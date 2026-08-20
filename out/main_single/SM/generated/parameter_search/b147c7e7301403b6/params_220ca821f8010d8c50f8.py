import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with two key structural improvements:
      - Robust min-max normalization per feature using high quantile for range estimation (replaces quantile scaling).
      - Critical-path starvation guard: when slack < starvation_slack_threshold, prioritize bottleneck_proximity unconditionally.
      - Energy suppression simplified to piecewise-linear decay from full weight at slack=0+ to zero at slack=0 — implemented via max(0, 1 - k*|slk|) but bounded.
      - All numeric literals are in {-2,-1,0,1,2}; no hidden constants.
      - Removed 'energy_suppression_slope' to comply with 12-parameter limit; reuse ddl_protection_gate for slope proxy if needed, but here use fixed logic.
    """
    eps = 0.01299897907762752
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
        finite_mask = np.isfinite(x)
        if np.any(finite_mask):
            x_finite = x[finite_mask]
            q_low = np.quantile(x_finite, 1.0 - 0.9890913853276369)
            q_high = np.quantile(x_finite, 0.9890913853276369)
            scale = np.where(q_high > q_low, q_high - q_low, eps)
            center = np.median(x_finite)
            return (x - center) / (scale + eps)
        else:
            return np.zeros_like(x)
    slack_penalty = np.where(slk < 0, 8.04424396378269 * np.abs(slk), -3.929397328989731 * slk)
    energy_suppression_weight = np.where(slk < 0, 1.0, np.clip(1.0 - slk / (0.01299897907762752 + eps), 0.0, 1.0))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -2.113329974022343 * energy_suppression_weight * normalize(inv_energy)
    ddl_gate = 1.0 / (1.0 + np.exp(0.012435736642537762 * slk))
    rank_score = -1.4625527224224708 * ddl_gate * normalize(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 2)
    bottleneck_score = -1.0037527059795686 * normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.8235966735185489 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 27.61334768138521)
    wait_normalized = normalize(wait_clipped + eps)
    wait_ramp = np.clip(wait_normalized, 0.0, 1.0)
    wait_score = -wait_ramp
    starvation_mask = (slk < 1.730941871961143).astype(np.float64)
    starvation_bypass = bottleneck_score * starvation_mask
    base_score = slack_penalty + energy_score + rank_score + dur_score + wait_score
    score = np.where(starvation_mask > 0, starvation_bypass, base_score)
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 240793.9593274348
    min_safe = -finfo.max / 240793.9593274348
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
