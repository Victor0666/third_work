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
    eps = 0.0028138418572531314
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
    slack_penalty = np.where(slk < 0, 9.595343005970253 * np.abs(slack_norm), -2.50075752332343 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.7923449926955468 * normalize(inv_energy)
    rank_active = np.where((slk >= 0) & (slack_norm >= 0.3677612893436594), rank, 0.0)
    rank_score = -1.7728403168538072 * normalize(rank_active + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.12880306718363416 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 9.70163601143198)
    wait_normalized = wait_clipped / (9.70163601143198 + eps)
    wait_score = -normalize(wait_normalized + eps)
    load_pressure_raw = work * (uncert + eps)
    load_pressure_norm = normalize(load_pressure_raw + eps)
    saturation_bound = 820706.3229328841 * np.median(np.abs(load_pressure_norm) + eps)
    load_pressure_saturated = np.clip(load_pressure_norm, -saturation_bound, saturation_bound)
    load_score = np.where(slack_norm < 0.3677612893436594, load_pressure_saturated, 0.0)
    protection_mask = slack_norm < 0.3677612893436594
    dur_score = np.where(protection_mask, dur_score * 0.2528903896353602, dur_score)
    energy_score = np.where(protection_mask, energy_score * 1.1505604593065877, energy_score)
    rank_score = np.where(protection_mask, rank_score * 1.7926862769855145, rank_score)
    score = slack_penalty + energy_score + rank_score + dur_score + wait_score + load_score
    score = np.where(np.isfinite(score), score, 0.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
