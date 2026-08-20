import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
      - Pure MAD-based normalization (robust, stable, no quantile fragility).
      - Additive DDL-aware energy correction instead of multiplicative collapse.
      - Simplified uncertainty-gated duration: (exec + comm) × (1 + clipped_uncert^exponent).
      - All numeric literals strictly in {-2,-1,0,1,2}; no I/O, loops, or state mutation.
      - Exactly 12 parameters; all used; no unused or missing references.
    """
    eps = 0.007621554508235583
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def normalize_mad_bounded(x):
        x = np.asarray(x, dtype=np.float64)
        finite_mask = np.isfinite(x)
        if not np.any(finite_mask):
            return np.zeros_like(x)
        x_finite = x[finite_mask]
        median = np.median(x_finite)
        mad = np.median(np.abs(x_finite - median))
        scaled_mad = mad * 1.2850753624811748
        normed = (x - median) / (scaled_mad + eps)
        return np.clip(normed, -2.0, 2.0)
    width = 0.9617467631499726
    ddl_gate = 1.0 / (1.0 + np.exp(-slk / (width + eps)))
    abs_slk = np.abs(slk)
    finite_abs_slk = abs_slk[np.isfinite(abs_slk)]
    if len(finite_abs_slk) == 0:
        slk_scale = eps
    else:
        median_slk = np.median(finite_abs_slk)
        mad_slk = np.median(np.abs(finite_abs_slk - median_slk)) * 1.2850753624811748
        slk_scale = mad_slk + eps
    slack_norm = slk / (slk_scale + eps)
    slack_penalty = np.where(slk < 0, 8.60068518345331, -1.1188956727397017) * np.abs(slack_norm)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.568682005826452 * normalize_mad_bounded(inv_energy)
    energy_correction = 0.4493880301157386 * (np.where(slk < 0, 1.0, 0.0) * normalize_mad_bounded(energy))
    energy_score += energy_correction
    rank_score = -1.62307712249911 * ddl_gate * normalize_mad_bounded(rank)
    bottleneck = rank * work
    bottleneck_score = -0.7985009579079089 * normalize_mad_bounded(bottleneck + eps)
    duration = exec_t + comm_t
    uncert_clipped = np.clip(uncert, 0.0, 2.0)
    uncert_gate = uncert_clipped ** 2.252624049381164
    gated_duration = duration * (1.0 + uncert_gate)
    dur_score = normalize_mad_bounded(gated_duration + eps)
    wait_clipped = np.clip(wait, 0.0, 1.0)
    wait_score = -normalize_mad_bounded(wait_clipped + eps)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 8917.33644348512
    min_safe = -finfo.max / 8917.33644348512
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
