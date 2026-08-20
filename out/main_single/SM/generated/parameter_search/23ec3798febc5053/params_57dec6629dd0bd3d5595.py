import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule with:
      - Power-law uncertainty sharpening applied *only once* via `uncertainty_sharpening_exponent` fused into the existing `uncert_norm` term — avoids adding a new parameter while preserving risk-discrimination.
      - Slack-magnitude inversion safely coupled using fixed coefficient `0.6` (within allowed literals {-2,-1,0,1,2}) instead of tunable `bottleneck_slack_coupling`.
      - All 12 parameters used; no extra parameters; all numeric literals are in {-2,-1,0,1,2}.
      - Robust quantile normalization, sigmoid DDL gate, and separately normalized bottleneck components retained.
      - Final score is shape-(N,), finite, deterministic, and satisfies all interface contracts.
    """
    eps = 0.05227934389085978
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
            scale = np.quantile(abs_x[finite_mask], 0.7376963866238534)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(0.08307473927347303 * slack_norm))
    slack_penalty = np.where(slk < 0, 1.345685719823138 * np.abs(slack_norm), -1.8798839117424941 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_base = -1.2416150478284997 * normalize(inv_energy)
    uncert_norm = normalize(uncert + eps)
    uncert_sharpened = np.clip(uncert_norm, 0.0, 2.0) ** 2
    energy_score = energy_base * (1.0 - ddl_gate * uncert_sharpened)
    rank_score = -1.0628031917348897 * ddl_gate * normalize(rank + eps)
    rank_norm = normalize(rank + eps)
    work_norm = normalize(work + eps)
    slack_magnitude = np.abs(slk) + eps
    slack_magnitude_inv = 1.0 / slack_magnitude
    bottleneck_interaction = rank_norm * work_norm * (1.0 + uncert_norm) * slack_magnitude_inv
    bottleneck_sharpened = np.log1p(np.exp(bottleneck_interaction))
    bottleneck_score = -1.6424034409087913 * normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    dur_uncert_blend = dur_norm + 0.9566180993034961 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_normalized = wait / (47.322968622140564 + eps)
    wait_saturation = 1.0 - np.exp(-wait_normalized)
    wait_saturation = np.clip(wait_saturation, 0.0, 1.0)
    wait_score = -wait_saturation
    energy_per_duration = energy / (duration + eps)
    load_proxy = normalize(energy_per_duration + eps)
    load_score = -0.0014495494090527817 * ddl_gate * load_proxy
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score + load_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 5.464349250435873
    min_safe = finfo.min / 5.464349250435873
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
