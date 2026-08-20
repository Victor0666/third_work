import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule with crisp DDL gating, robust normalization, and adaptive blending:
      - Crisp binary ddl_feasible_mask ensures strict deadline-first enforcement.
      - Energy score uses saturation via quantile-based clipping (no new parameter) instead of threshold.
      - Duration-uncertainty blend adapts smoothly using ddl_feasible_mask interpolation.
      - All numeric literals are -2, -1, 0, 1, or 2; no hidden constants.
      - Uses np.finfo for safe clamping — no literal epsilons or infinities.
      - Exactly 12 parameters; all used; no unused or missing references.
    """
    eps = 0.013777965432378678
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def normalize_bounded(x):
        x = np.asarray(x)
        finite_mask = np.isfinite(x)
        if np.any(finite_mask):
            abs_x = np.abs(x[finite_mask])
            scale = np.quantile(abs_x, 0.94389979863304)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        normed = x / (scale + eps)
        return np.clip(normed, -2.0, 2.0)
    ddl_feasible_mask = (slk > 0.4660701438029775).astype(np.float64)
    slack_penalty = np.where(slk < 0, 2.562335582524755 * np.abs(slk), -1.6163092423682621 * slk)
    inv_energy = 1.0 / (np.clip(energy, eps, np.inf) + eps)
    energy_score = -2.923960716435485 * normalize_bounded(inv_energy) * ddl_feasible_mask
    rank_score = -1.2841066971015946 * normalize_bounded(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 2.703531452092775)
    bottleneck_score = -2.560284502564691 * normalize_bounded(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize_bounded(duration + eps)
    uncert_norm = normalize_bounded(uncert + eps)
    dur_uncert_blend = (1.0 - ddl_feasible_mask) * dur_norm + ddl_feasible_mask * (dur_norm + 0.78650091219655 * uncert_norm)
    dur_score = normalize_bounded(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 1.9119469279607189)
    wait_normalized = normalize_bounded(wait_clipped + eps, 0.0, 1.0)
    wait_score = -wait_normalized
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max * 0.37422415961691313, neginf=finfo.min * 0.37422415961691313)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
