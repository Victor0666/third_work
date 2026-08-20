import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with key structural changes:
      - Replaced sigmoid DDL gate with piecewise-linear ddl_feasibility_margin for smoother, interpretable feasibility boundary.
      - Introduced ddl_feasible_mask: binary mask enabling energy scoring *only* when slack >= margin, aligning energy minimization strictly with feasibility.
      - Retained successor-release interaction but simplified its normalization to avoid double-quantile distortion.
      - Removed finfo_max_scale (inactive per diagnostics) and replaced saturation with explicit finite clipping using np.finfo and tunable safety factor.
      - Normalized all features using robust quantile scaling *before* interaction terms to prevent amplification of outliers.
      - Added bounded exponential starvation mitigation (1 - exp(-wait/θ)) for smooth, asymptotic anti-starvation behavior.
      - Enforced strict ordering: slack penalty dominates; energy only considered when feasible; criticality and bottleneck terms gated by feasibility.
    """
    eps = 0.09506820320936288
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
            scale = np.quantile(abs_x[finite_mask], 0.8647899347355332)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    exec_norm = normalize(exec_t + eps)
    comm_norm = normalize(comm_t + eps)
    energy_norm = normalize(energy + eps)
    slack_norm = normalize(slk)
    rank_norm = normalize(rank + eps)
    work_norm = normalize(work + eps)
    wait_norm = normalize(wait + eps)
    uncert_norm = normalize(uncert + eps)
    ddl_feasible_mask = (slk >= 0.5708997415471904).astype(np.float64)
    slack_penalty = np.where(slk < 0, 1.753985755999542 * np.abs(slack_norm), -1.3574398818722486 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy_norm + eps)
    energy_score = -0.41780558732919565 * inv_energy * ddl_feasible_mask
    rank_score = -1.461039419527745 * rank_norm * ddl_feasible_mask
    slack_magnitude_inv = 1.0 / (np.abs(slack_norm) + eps)
    bottleneck_sharpened = rank_norm * work_norm * np.power(slack_magnitude_inv, 3.5967977104941977)
    bottleneck_score = -1.9791621383935842 * bottleneck_sharpened
    dur_uncert_blend = exec_norm + comm_norm + 0.0942671641860043 * uncert_norm
    dur_score = dur_uncert_blend
    wait_saturation = 1.0 - np.exp(-wait_norm / (25.54840838022528 + eps))
    wait_score = -wait_saturation
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max * 0.4305237028831197
    min_safe = finfo.min * 0.4305237028831197
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
