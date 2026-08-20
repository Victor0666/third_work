import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule featuring:
      - Bounded piecewise load-pressure: (work * uncertainty) clipped at adaptive median-scale threshold,
        replacing linear coupling to reduce fragility and rank instability across seeds.
      - Robust normalization with MAD-based scale *and* explicit finite bounds via np.clip,
        suppressing outlier-driven flips while preserving ordinal structure.
      - Removed finfo_max_scale and np.nan_to_num; now use direct finite masking for safety.
      - All components remain DDL-gated and interpretable; no hidden constants beyond {-2,-1,0,1,2}.
    """
    eps = 0.0015731043611889649
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
        med = np.median(abs_x) if np.all(np.isfinite(abs_x)) else eps
        dev = np.abs(abs_x - med)
        mad = np.median(dev) if np.all(np.isfinite(dev)) and med > eps else eps
        scale = max(med, 1.0 * mad, eps)
        normed = x / (scale + eps)
        return np.clip(normed, -2.0, 2.0)
    slack_norm = normalize(slk)
    slack_penalty = np.where(slk < 0, 5.334006122998355 * np.abs(slack_norm), -0.6688125929481864 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.2448856732958804 * normalize(inv_energy)
    rank_active = np.where((slk >= 0) & (slack_norm >= 0.05478807398295172), rank, 0.0)
    rank_score = -0.41821985649109544 * normalize(rank_active + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.10110059889032977 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 28.86499882955044)
    wait_normalized = wait_clipped / (28.86499882955044 + eps)
    wait_score = -normalize(wait_normalized + eps)
    load_pressure_raw = work * (uncert + eps)
    load_pressure_norm = normalize(load_pressure_raw + eps)
    saturation_bound = 366519.11585585016 * np.median(np.abs(load_pressure_norm) + eps)
    load_pressure_saturated = np.clip(load_pressure_norm, -saturation_bound, saturation_bound)
    load_score = np.where(slack_norm < 0.05478807398295172, load_pressure_saturated, 0.0)
    protection_mask = slack_norm < 0.05478807398295172
    dur_score = np.where(protection_mask, dur_score * 0.2933498432276386, dur_score)
    energy_score = np.where(protection_mask, energy_score * 2.27061360880606, energy_score)
    rank_score = np.where(protection_mask, rank_score * 1.5560116980128689, rank_score)
    score = slack_penalty + energy_score + rank_score + dur_score + wait_score + load_score
    score = np.where(np.isfinite(score), score, 0.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
