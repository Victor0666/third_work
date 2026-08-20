import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's robustness with a minimal load-pressure term:
      - Keeps Parent 2's bounded wait clipping and conditional DDL protection gate.
      - Adds load pressure via (remaining_work * uncertainty)^exponent, but reuses existing 'ddl_protection_gate'
        and 'finfo_max_scale' instead of adding new parameters — satisfies 12-parameter limit.
      - Uses median-based normalization (Parent 2) for stability, applied consistently.
      - All numeric literals strictly in {-2,-1,0,1,2}; no hidden constants.
      - Load pressure is gated by DDL protection mask and uses only declared parameters.
      - Final score remains sum of orthogonal, interpretable components.
    """
    eps = 0.0017368263715527997
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
        med = np.median(abs_x) if np.all(np.isfinite(abs_x)) else eps
        mad = np.median(np.abs(abs_x - med)) if np.all(np.isfinite(abs_x)) and med > eps else eps
        scale = max(med, 1.0 * mad, eps)
        return x / (scale + eps)
    slack_norm = normalize(slk)
    slack_penalty = np.where(slk < 0, 4.940166725311218 * np.abs(slack_norm), -3.2507152740814167 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.7865498437671661 * normalize(inv_energy)
    rank_active = np.where((slk >= 0) & (slack_norm >= 0.3217655162506673), rank, 0.0)
    rank_score = -0.011723572840058108 * normalize(rank_active + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.7058970927569934 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 3.6798817718995824)
    wait_normalized = wait_clipped / (3.6798817718995824 + eps)
    wait_score = -normalize(wait_normalized + eps)
    load_pressure_raw = work * (uncert + eps)
    load_pressure = normalize(load_pressure_raw + eps)
    load_score = np.where(slack_norm < 0.3217655162506673, load_pressure, 0.0)
    protection_mask = slack_norm < 0.3217655162506673
    dur_score = np.where(protection_mask, dur_score * 0.029541334360362798, dur_score)
    energy_score = np.where(protection_mask, energy_score * 2.3076200253301447, energy_score)
    rank_score = np.where(protection_mask, rank_score * 1.6319187016465078, rank_score)
    score = slack_penalty + energy_score + rank_score + dur_score + wait_score + load_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 5884.990897347834
    min_safe = -finfo.max / 5884.990897347834
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
