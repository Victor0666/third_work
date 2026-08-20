import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with key structural changes:
      - Replaced sigmoid DDL gate with piecewise-linear feasibility margin (more interpretable, less fragile)
      - Introduced ddl_feasible_mask to conditionally enable energy scoring only when slack > margin
      - Removed unused ddl_protection_gate parameter
      - Added robust clipping for all normalized features to [-2, 2] to prevent rank inversions
      - Kept successor-release interaction but re-normalized via absolute quantile for signed slack terms
      - Starvation term now uses bounded exponential saturation (1 - exp(-wait/theta)) for smoother long-tail response
      - All feature interactions remain bounded, deterministic, and free of infinite loops/divisions
    """
    eps = 0.0022452175284757075
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def normalize_signed(x):
        x = np.asarray(x)
        finite_mask = np.isfinite(x)
        if np.any(finite_mask):
            scale = np.quantile(np.abs(x[finite_mask]), 0.6766938540850908)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        normed = x / (scale + eps)
        return np.clip(normed, -2.0, 2.0)

    def normalize_positive(x):
        x = np.asarray(x)
        finite_mask = np.isfinite(x) & (x >= 0)
        if np.any(finite_mask):
            scale = np.quantile(x[finite_mask], 0.6766938540850908)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        normed = x / (scale + eps)
        return np.clip(normed, 0.0, 2.0)
    slack_norm = normalize_signed(slk)
    ddl_feasibility_margin = 0.5021715598079122
    ddl_feasible_mask = np.where(slk > ddl_feasibility_margin, 1.0, np.clip((slk - ddl_feasibility_margin) / (eps + ddl_feasibility_margin), 0.0, 1.0))
    slack_penalty = np.where(slk < 0, 4.213723134854133 * np.abs(slack_norm), -0.8159012073282893 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.8978401628630062 * normalize_positive(inv_energy) * ddl_feasible_mask
    rank_score = -2.384288129840848 * normalize_positive(rank + eps) * ddl_feasible_mask
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 0.7052054160606525)
    bottleneck_score = -3.5788056078665598 * normalize_positive(bottleneck_sharpened + eps) * ddl_feasible_mask
    duration = exec_t + comm_t
    dur_norm = normalize_positive(duration + eps)
    uncert_norm = normalize_positive(uncert + eps)
    dur_uncert_blend = dur_norm + 1.5162799715886461 * uncert_norm
    dur_score = normalize_positive(dur_uncert_blend)
    wait_saturation = 1.0 - np.exp(-wait / (20.74550924245171 + eps))
    wait_score = -np.clip(wait_saturation, 0.0, 1.0)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 21378355.080063786
    min_safe = -finfo.max / 21378355.080063786
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
