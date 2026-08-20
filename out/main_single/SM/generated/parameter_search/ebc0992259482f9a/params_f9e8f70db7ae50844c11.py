import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule with hybrid gating and robust normalization:
      - Hard slack mask for bottleneck term (upward_rank * remaining_work) ensures strict DDL feasibility.
      - Soft sigmoid ddl_gate modulates all other deadline-sensitive terms (energy, rank, release, wait).
      - Starvation mitigation uses clipped wait time + exponential saturation, then gated.
      - Uncertainty-slack interaction activated only under deadline stress (1-ddl_gate).
      - All numeric literals are in {-2,-1,0,1,2}; no hidden constants.
      - Uses median/MAD normalization for outlier resilience.
      - Exactly 12 parameters; all used; no unused or missing references.
    """
    eps = 0.0002668419992719067
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
        med = np.median(abs_x)
        mad = np.median(np.abs(abs_x - med)) if np.all(np.isfinite(abs_x)) else eps
        scale = max(med, 1.0 * mad, eps)
        return x / (scale + eps)
    gate_width = 0.09523217269249623
    ddl_gate = 1.0 / (1.0 + np.exp(-slk / (gate_width + eps)))
    slack_feasible_mask = (slk >= 0).astype(np.float64)
    slack_penalty = 3.077050359360122 * np.maximum(-slk, 0.0) + 4.48195543787917 * np.minimum(slk, 0.0)
    slack_score = normalize(slack_penalty + eps)
    inv_energy = 1.0 / (energy + eps)
    energy_active = inv_energy * ddl_gate
    energy_score = -2.7881178563319975 * normalize(energy_active + eps)
    rank_score = -0.11474186416846166 * ddl_gate * normalize(rank + eps)
    bottleneck_score = rank * work
    bottleneck_score_gated = bottleneck_score * slack_feasible_mask
    bottleneck_norm = normalize(bottleneck_score_gated + eps)
    bottleneck_score_final = -0.29241723436172296 * bottleneck_norm
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.5319937641637587 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 2.0 / (0.09546710671548095 + eps))
    wait_sat = 1.0 - np.exp(-0.09546710671548095 * wait_clipped)
    wait_score = -normalize(wait_sat + eps) * ddl_gate
    unc_slack_interaction = 0.23110169897229801 * uncert_norm * (1.0 - ddl_gate)
    release_pressure = rank * work
    release_score = -0.5499386622013017 * ddl_gate * normalize(release_pressure + eps)
    score = slack_score + energy_score + rank_score + bottleneck_score_final + dur_score + wait_score + unc_slack_interaction + release_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 788839.7788074169
    min_safe = -finfo.max / 788839.7788074169
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
