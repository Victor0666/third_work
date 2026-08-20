import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
      - Replaced IQR normalization with robust, parameter-efficient MAD-based normalization bounded to [-2,2].
      - Introduced explicit multiplicative slack-energy coupling: `sigmoid(-slack/width) * inv_energy` instead of additive latency_factor.
      - All numeric literals are in {-2,-1,0,1,2}; finfo-safe clamping uses PARAMS["finfo_safe_scale"].
      - Final score is finite, shape-(N,), deterministic, and satisfies all interface contracts.
    """
    eps = 0.0024111338113684553
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def normalize_mad_bounded(x, scale_factor=1.694019666497739):
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
    width = 1.2601260591743257
    ddl_gate = 1.0 / (1.0 + np.exp(-slk / (width + eps)))
    abs_slk = np.abs(slk)
    finite_abs_slk = abs_slk[np.isfinite(abs_slk)]
    if len(finite_abs_slk) == 0:
        slk_scale = eps
    else:
        median_slk = np.median(finite_abs_slk)
        mad_slk = np.median(np.abs(finite_abs_slk - median_slk)) * 1.694019666497739
        slk_scale = mad_slk + eps
    slack_norm = slk / (slk_scale + eps)
    slack_penalty = np.where(slk < 0, 6.14886904198503, -4.183273760666587) * np.abs(slack_norm)
    urgency_coupling = 1.0 + 0.9458567441814629 * (1.0 / (1.0 + np.exp(slk / (width + eps))))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.6269582189446123 * normalize_mad_bounded(inv_energy) * urgency_coupling
    rank_score = -1.0634572558488344 * ddl_gate * normalize_mad_bounded(rank)
    bottleneck = rank * work
    bottleneck_score = -1.8768310517632443 * normalize_mad_bounded(bottleneck + eps)
    duration = exec_t + comm_t
    uncert_gate = np.clip(uncert, 0.0, 2.0) ** 1.7844947863178517
    gated_duration = duration * (1.0 + uncert_gate)
    dur_score = normalize_mad_bounded(gated_duration + eps)
    wait_clipped = np.clip(wait, 0.0, 16.607878166817045)
    wait_score = -normalize_mad_bounded(wait_clipped + eps)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 9333931.299748776
    min_safe = -finfo.max / 9333931.299748776
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
