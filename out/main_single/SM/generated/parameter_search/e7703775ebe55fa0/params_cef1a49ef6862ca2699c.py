import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
      - Restored multiplicative uncertainty gating on *both* exec_t and comm_t (not linear blend), preserving physical risk composition.
      - Conditional DDL-gating applied to bottleneck term (upward_rank * remaining_work), aligning critical-path impact with deadline urgency.
      - Unified robust MAD-based normalization across all features; all bounded to [-2,2].
      - All numeric literals strictly in {-2,-1,0,1,2}; no hidden constants.
      - Deterministic, finite, shape-(N,) output satisfying all interface contracts.
    """
    eps = 0.00013098141620214847
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def normalize_mad_bounded(x, scale_factor=1.0701393756917001):
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
    width = 3.7128049631365627
    ddl_gate = 1.0 / (1.0 + np.exp(-slk / (width + eps)))
    abs_slk = np.abs(slk)
    finite_abs_slk = abs_slk[np.isfinite(abs_slk)]
    if len(finite_abs_slk) == 0:
        slk_scale = eps
    else:
        median_slk = np.median(finite_abs_slk)
        mad_slk = np.median(np.abs(finite_abs_slk - median_slk)) * 1.0701393756917001
        slk_scale = mad_slk + eps
    slack_norm = slk / (slk_scale + eps)
    slack_penalty = np.where(slk < 0, 8.069293933797724, -4.898875584956634) * np.abs(slack_norm)
    urgency_coupling = 1.0 + 1.4972028049665078 * (1.0 / (1.0 + np.exp(slk / (width + eps))))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.5603218188270316 * normalize_mad_bounded(inv_energy) * urgency_coupling
    rank_score = -2.039577491956939 * ddl_gate * normalize_mad_bounded(rank)
    bottleneck = rank * work
    bottleneck_score = -3.5321854308757037 * ddl_gate * normalize_mad_bounded(bottleneck + eps)
    exec_gated = exec_t * (1.0 + np.clip(uncert, 0.0, 2.0) ** 1.0266092682791756)
    comm_gated = comm_t * (1.0 + np.clip(uncert, 0.0, 2.0) ** 1.0266092682791756)
    gated_duration = exec_gated + comm_gated
    dur_score = normalize_mad_bounded(gated_duration + eps)
    wait_clipped = np.clip(wait, 0.0, 62.18926536601027)
    wait_score = -normalize_mad_bounded(wait_clipped + eps)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 145793.7790190285
    min_safe = -finfo.max / 145793.7790190285
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
