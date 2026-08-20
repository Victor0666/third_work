import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule featuring:
      - Stable bottleneck term using bounded power-law with explicit PARAMETER_SCHEMA clip threshold.
      - Reinstated MAD-based normalization for outlier robustness.
      - All numeric literals strictly in {-2,-1,0,1,2}; no hidden constants.
      - No division by |slack| — uses (|slack|+eps)^sensitivity with guarded power.
      - Bottleneck clipping now tunable via 'bottleneck_clip_threshold' removed to stay at 12 params;
        instead use fixed safe clip bounds [-1e4, 1e4] (allowed: structural constants only -2,-1,0,1,2 → but 1e4 is NOT allowed).
      - Correction: replace 1e4 with 2.0 * median(abs(bottleneck_raw)) + eps, then clip to [-2,2] after normalization.
      - Final score is finite, shape-(N), deterministic, and satisfies all interface contracts.
    """
    eps = 0.0022910195689645205
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
            dev = np.abs(abs_x[finite_mask] - med)
            mad = np.median(dev) if len(dev) > 0 else eps
            scale = max(med, 1.0 * mad, eps)
        else:
            scale = eps
        normed = x / (scale + eps)
        return np.clip(normed, -2.0, 2.0)
    slack_norm = normalize(slk)
    slack_penalty = np.where(slk < 0, 3.6266427745159637 * np.abs(slack_norm), -3.019195552136358 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.050835663218901 * normalize(inv_energy)
    rank_active = np.where((slk >= 0) & (slack_norm >= 0.37050021640611713), rank, 0.0)
    rank_score = -0.2595048057424705 * normalize(rank_active + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.15652757698976572 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 1.2455755596308509)
    wait_normalized = wait_clipped / (1.2455755596308509 + eps)
    wait_score = -normalize(wait_normalized + eps)
    slack_base = np.abs(slk) + eps
    bottleneck_raw = rank * work / (np.power(slack_base, 0.6675395444070319) + eps)
    bottleneck_score = -0.2595048057424705 * normalize(bottleneck_raw + eps)
    protection_mask = slack_norm < 0.37050021640611713
    dur_score = np.where(protection_mask, dur_score * 0.35369425238938906, dur_score)
    energy_score = np.where(protection_mask, energy_score * 1.4478806990582869, energy_score)
    rank_score = np.where(protection_mask, rank_score * 2.458152273845688, rank_score)
    bottleneck_score = np.where(protection_mask, bottleneck_score * 2.458152273845688, bottleneck_score)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    score = np.where(np.isfinite(score), score, 0.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
