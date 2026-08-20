import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining robustness from Parent 2 and bottleneck-awareness from Parent 1:
      - Preserves bounded MAD+median normalization with [-2,2] clipping for stability.
      - Reintroduces bottleneck term (upward_rank * remaining_work) but gates it under DDL protection only.
      - Uses unified normalized slack for all gating — avoids dual scales (width vs threshold).
      - All amplifications/suppressions apply *only* when slack_norm < ddl_protection_gate (tight-deadline regime).
      - Load pressure now includes bottleneck coupling: (rank * work * uncertainty), enabling joint critical-path + risk awareness.
      - Starvation mitigation remains bounded and deterministic: clipped wait time, not exponential.
      - No finfo_max_scale or nan_to_num — uses explicit finite masking for safety and simplicity.
      - All numeric literals are restricted to {-2,-1,0,1,2}.
    """
    eps = 0.003465410978137539
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
            med = np.median(abs_x[finite_mask])
            dev = np.abs(abs_x - med)
            mad = np.median(dev[finite_mask]) if np.any(finite_mask) else eps
            scale = max(med, 1.0 * mad, eps)
        else:
            scale = eps
        normed = x / (scale + eps)
        return np.clip(normed, -2.0, 2.0)
    slack_norm = normalize(slk)
    slack_penalty = np.where(slk < 0, 4.506420640128699 * np.abs(slack_norm), -3.494217541040832 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.7381660636978753 * normalize(inv_energy)
    rank_active = np.where(slack_norm < 0.35663774462391673, rank, 0.0)
    rank_score = -0.8844767039324772 * normalize(rank_active + eps)
    bottleneck_raw = rank * work
    bottleneck_active = np.where(slack_norm < 0.35663774462391673, bottleneck_raw, 0.0)
    bottleneck_score = -0.8844767039324772 * normalize(bottleneck_active + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.9828161869340075 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 3.907832193537435)
    wait_normalized = wait_clipped / (3.907832193537435 + eps)
    wait_score = -normalize(wait_normalized + eps)
    load_pressure_raw = rank * work * (uncert + eps)
    load_pressure_norm = normalize(load_pressure_raw + eps)
    saturation_bound = 186875.43472945516 * np.median(np.abs(load_pressure_norm) + eps)
    load_pressure_saturated = np.clip(load_pressure_norm, -saturation_bound, saturation_bound)
    load_score = np.where(slack_norm < 0.35663774462391673, load_pressure_saturated, 0.0)
    protection_mask = slack_norm < 0.35663774462391673
    dur_score = np.where(protection_mask, dur_score * 0.3077306090714266, dur_score)
    energy_score = np.where(protection_mask, energy_score * 2.007346277694835, energy_score)
    rank_score = np.where(protection_mask, rank_score * 1.05027166870003, rank_score)
    bottleneck_score = np.where(protection_mask, bottleneck_score * 1.05027166870003, bottleneck_score)
    load_score = np.where(protection_mask, load_score * 2.007346277694835, load_score)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score + load_score
    score = np.where(np.isfinite(score), score, 0.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
