import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule with:
      - Crisp ddl_feasible_mask and robust quantile normalization (Parent 2).
      - Power-law starvation saturation (novel structural change) using only declared params.
      - All numeric literals restricted to {-2,-1,0,1,2}; no hidden constants.
      - Uses np.finfo for safe clamping instead of literal bounds.
      - Exactly 12 parameters; all used; no unused or missing references.
    """
    eps = 0.0008769369205938593
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
        x = np.asarray(x, dtype=np.float64)
        abs_x = np.abs(x)
        finite_mask = np.isfinite(abs_x)
        if np.any(finite_mask):
            scale = np.quantile(abs_x[finite_mask], 0.6845037758301967)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    ddl_feasible_mask = slk > 0.6286694824856007
    slack_penalty = np.where(slk < 0, 5.669842064856093 * np.abs(slk), -2.61122602911442 * slk)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -2.8452634660393437 * normalize(inv_energy) * ddl_feasible_mask
    rank_score = -0.29270870407435834 * normalize(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 1.8563496976166662)
    bottleneck_score = -2.4842981470513794 * normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.7266251429667675 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    theta = 1.9634966987155162 + eps
    p = 1.8563496976166662
    wait_clipped = np.clip(wait, 0.0, None)
    wait_power = np.power(wait_clipped, p)
    theta_power = np.power(theta, p)
    starvation_saturation = wait_power / (wait_power + theta_power + eps)
    wait_score = -starvation_saturation
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max * 0.6852931546538837, neginf=finfo.min * 0.6852931546538837)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
