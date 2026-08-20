import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with three structural improvements:
      - Added host-load-aware energy suppression: energy optimization only activated when DDL safety margin exceeds threshold
      - Replaced linear wait ramp with smooth power-law saturation: preserves monotonicity while reducing sensitivity to outliers
      - Introduced piecewise-linear robust slack gating: replaces sigmoid with interpretable, bounded transition around slack=0
      - Removed finfo_max_scale (inactive per diagnostics) and simplified normalization safeguards
      - All features now strictly normalized using quantile scaling with epsilon fallback
      - Critical-path terms retain dominance via slack penalty + bottleneck interaction
    """
    eps = 0.00023054437020171242
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
            scale = np.quantile(abs_x[finite_mask], 0.8648526244631246)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    ddl_gate_width = 0.001959376599618131
    midpoint = 0.948503470179804
    ddl_gate = np.clip(midpoint + midpoint * (slk / (ddl_gate_width + eps)), 0.0, 1.0)
    slack_penalty = np.where(slk < 0, 2.789322418518721 * np.abs(slk), -2.225328348154687 * slk)
    safe_margin = slk - (exec_t + comm_t)
    energy_active = (safe_margin > eps).astype(np.float64)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.5253853921888572 * normalize(inv_energy) * energy_active
    load_proxy = normalize(uncert + eps) + normalize(exec_t + eps)
    load_suppression = 1.0 - 0.18325597956943432 * np.clip(load_proxy, 0.0, 1.0)
    energy_score = energy_score * load_suppression
    rank_score = -0.7237439761256415 * ddl_gate * normalize(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 2.6210351530046587)
    bottleneck_score = -1.1321426898924631 * normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.6905950864603139 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_normalized = normalize(wait + eps)
    wait_saturation = np.power(np.clip(wait_normalized, 0.0, 1.0), 0.18325597956943432)
    wait_score = -wait_saturation
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
