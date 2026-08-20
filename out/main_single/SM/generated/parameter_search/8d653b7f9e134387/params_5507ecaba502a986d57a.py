import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with 12 parameters:
      - Removes 'smooth_load_pressure_weight' to comply with 12-parameter limit.
      - Replaces brittle median-based load-pressure clipping with a *robustly normalized* and *fully ddl-gated* version of 'work * uncert',
        but now integrated into the existing bottleneck term via multiplicative coupling — no new parameter needed.
      - Bottleneck term is enhanced: upward_rank * remaining_work * (1 + uncertainty) * slack_magnitude_inv, all quantile-normalized.
      - This preserves structural expressivity while eliminating unbounded interactions and reducing parameter count.
      - All numeric literals are -2, -1, 0, 1, or 2; epsilon and finfo handled via PARAMS and np.finfo.
      - Strictly enforces finite output, shape (N,), and deterministic behavior.
    """
    eps = 0.001176863893978242
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
            scale = np.quantile(abs_x[finite_mask], 0.8054715636990162)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(0.16378457000798652 * slack_norm))
    slack_penalty = np.where(slk < 0, 7.287087984689291 * np.abs(slack_norm), -2.90926481317688 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -3.483306670574332 * normalize(inv_energy)
    rank_score = -1.3333731058961567 * ddl_gate * normalize(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_base = rank * work * (1.0 + uncert) * slack_magnitude_inv
    bottleneck_sharpened = np.log1p(np.exp(bottleneck_base))
    bottleneck_score = -2.4730749259311344 * normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.9896596785275764 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_normalized = wait / (33.845639055242785 + eps)
    wait_saturation = 1.0 - np.exp(-wait_normalized)
    wait_saturation = np.clip(wait_saturation, 0.0, 1.0)
    wait_score = -wait_saturation
    energy_per_duration = energy / (duration + eps)
    load_proxy = normalize(energy_per_duration + eps)
    load_score = -1.8765765695419785 * ddl_gate * load_proxy
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score + load_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 1.0283045243201816
    min_safe = finfo.min / 1.0283045243201816
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
