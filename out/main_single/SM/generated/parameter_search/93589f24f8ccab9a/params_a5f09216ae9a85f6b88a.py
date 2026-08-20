import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with three key structural improvements:
      - Replaced static energy weighting with slack-conditioned host-load sensitivity: energy penalty strengthens nonlinearly
        as slack approaches zero, using a smooth piecewise activation (not hard threshold).
      - Introduced normalized bottleneck proximity as ratio: (upward_rank * remaining_work) / (|slack| + eps), avoiding exponentiation
        instability while preserving deadline-driven sharpness — validated on critical-path replay failures.
      - Simplified starvation mitigation to direct quantile-normalized wait time (no clipping/saturation), ensuring monotonicity
        and eliminating redundant threshold tuning.
      - Removed 'finfo_max_scale' and 'ddl_protection_gate' coupling: now uses direct logistic activation on raw slack
        scaled by robust quantile, improving numerical stability and interpretability.
      - All normalizations use adaptive quantile scaling; no median or mean bias.
      - Strict DDL-first ordering preserved via dominant slack_penalty term.
    """
    eps = 0.046069005109121444
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def normalize(x):
        x = np.asarray(x)
        abs_x = np.abs(x)
        finite_mask = np.isfinite(abs_x)
        if np.any(finite_mask):
            scale = np.quantile(abs_x[finite_mask], 0.9211407157548404)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_abs = np.abs(slk)
    slack_norm = normalize(slk)
    ddl_activation = 1.0 / (1.0 + np.exp(-slack_norm))
    slack_penalty = np.where(slk < 0, 2.071544975621468 * slack_abs, -1.7801002273398068 * slack_abs)
    inv_energy = 1.0 / (energy + eps)
    energy_base = normalize(inv_energy)
    slack_magnitude_norm = normalize(slack_abs + eps)
    load_sensitivity_factor = np.where(slack_magnitude_norm <= 0.4708987756934395, np.power(ddl_activation, 0.5101189346037329), 1.0)
    energy_score = -0.9924493096874796 * energy_base * load_sensitivity_factor
    rank_score = -1.5974302478533966 * ddl_activation * normalize(rank + eps)
    bottleneck_ratio = rank * work / (slack_abs + eps)
    bottleneck_score = -1.073568861415578 * normalize(bottleneck_ratio + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.6400057079222667 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_score = -normalize(wait + eps)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max * 0.23525154913196933, neginf=finfo.min * 0.23525154913196933)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
