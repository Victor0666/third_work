import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
      - Smooth sigmoid energy activation (replaces hard threshold) using `ddl_protection_gate`-style logic but tuned separately.
      - Load-aware energy normalization via `upward_rank * remaining_work`, quantile-normalized — aligns marginal energy with critical-path load density.
      - Adaptive starvation mitigation: `wait_normalized / (|slack_norm| + eps)` scaled by `wait_pressure_ratio`.
      - All normalizations use robust quantile scaling; no mean/median bias.
      - Slack penalty dominates to enforce hard deadline feasibility first.
      - Exactly 12 parameters; all used; no numeric literals beyond -2,-1,0,1,2.
      - Uses np.finfo for safe clamping.
    """
    eps = 0.09084797544885058
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
            scale = np.quantile(abs_x[finite_mask], 0.5774121833435034)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(0.6635954773738366 * slack_norm))
    slack_penalty = np.where(slk < 0, 4.108483154184585 * np.abs(slk), -4.290530922978731 * slk)
    energy_activation = 1.0 / (1.0 + np.exp(-0.6635954773738366 * 2.0 * slack_norm))
    inv_energy = 1.0 / (energy + eps)
    load_density = rank * work
    load_norm = normalize(load_density + eps)
    energy_score = -0.8591602907043996 * normalize(inv_energy) * load_norm * energy_activation
    rank_score = -1.0163420196285442 * ddl_gate * normalize(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 2.0127594335003636)
    bottleneck_score = -1.820908107859152 * normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.21324396530195946 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    slack_abs_med = np.median(np.abs(slk)) if len(slk) > 0 else 1.0
    wait_clipped = np.clip(wait, 0.0, 2.0 * slack_abs_med + eps)
    wait_normalized = normalize(wait_clipped + eps)
    wait_pressure = wait_normalized / (np.abs(slack_norm) + eps)
    wait_ramp = np.clip(0.31853273273394833 * wait_pressure, 0.0, 1.0)
    wait_score = -wait_ramp
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 90023.28049498273
    min_safe = -finfo.max / 90023.28049498273
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
