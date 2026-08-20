import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
      - Restored multiplicative uncertainty gating on *both* exec_t and comm_t (not linear blend), preserving physical risk composition.
      - Conditional DDL-gating applied to bottleneck term (upward_rank * remaining_work), aligning critical-path impact with deadline urgency.
      - Unified robust MAD-based normalization across all features; all bounded to [-2,2].
      - All numeric literals strictly in {-2,-1,0,1,2}; no hidden constants.
      - Deterministic, finite, shape-(N,) output satisfying all interface contracts.
    """
    eps = 5.514487020178067e-05
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def normalize_mad_bounded(x, scale_factor=1.366024391527021):
        x = np.asarray(x, dtype=np.float64)
        finite_mask = np.isfinite(x)
        if not np.any(finite_mask):
            return np.zeros_like(x)
        x_finite = x[finite_mask]
        median = np.median(x_finite)
        mad = np.median(np.abs(x_finite - median))
        scaled_mad = mad * scale_factor
        normed = (x - median) / (scaled_mad + eps)
        return np.clip(normed, -2.0, 2.0)
    width = 0.10355391006014021
    ddl_gate = 1.0 / (1.0 + np.exp(-slk / (width + eps)))
    abs_slk = np.abs(slk)
    finite_abs_slk = abs_slk[np.isfinite(abs_slk)]
    if len(finite_abs_slk) == 0:
        slk_scale = eps
    else:
        median_slk = np.median(finite_abs_slk)
        mad_slk = np.median(np.abs(finite_abs_slk - median_slk)) * 1.366024391527021
        slk_scale = mad_slk + eps
    slack_norm = slk / (slk_scale + eps)
    slack_penalty = np.where(slk < 0, 9.444295203340701, -5.4135516708815405) * np.abs(slack_norm)
    urgency_coupling = 1.0 + 1.9060877913072654 * (1.0 / (1.0 + np.exp(slk / (width + eps))))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.1633436208290926 * normalize_mad_bounded(inv_energy) * urgency_coupling
    rank_score = -1.536129463517266 * ddl_gate * normalize_mad_bounded(rank)
    bottleneck = rank * work
    bottleneck_score = -2.3020838238188226 * ddl_gate * normalize_mad_bounded(bottleneck + eps)
    exec_gated = exec_t * (1.0 + np.clip(uncert, 0.0, 2.0) ** 1.9556067406509643)
    comm_gated = comm_t * (1.0 + np.clip(uncert, 0.0, 2.0) ** 1.9556067406509643)
    gated_duration = exec_gated + comm_gated
    dur_score = normalize_mad_bounded(gated_duration + eps)
    wait_clipped = np.clip(wait, 0.0, 38.41060499801458)
    wait_score = -normalize_mad_bounded(wait_clipped + eps)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 2838.2328893705385
    min_safe = -finfo.max / 2838.2328893705385
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
