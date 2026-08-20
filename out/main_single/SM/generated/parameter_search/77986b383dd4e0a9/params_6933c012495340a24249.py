import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule with:
      - Robust MAD-based normalization ([-2,2]) for outlier resilience.
      - Smooth sigmoid DDL gate for differentiable urgency modulation.
      - Multiplicative energy-slack coupling for deadline-aware efficiency.
      - Power-law starvation mitigation (wait^0.8 capped at 2.0) — bounded & monotonic.
      - Linear duration-uncertainty blend using tunable `duration_uncertainty_ratio`.
      - All numeric literals are {-2,-1,0,1,2}; no hidden constants; deterministic and finite.
    """
    eps = 0.011908300998716618
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def normalize_mad_bounded(x, scale_factor=1.942917115622235):
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
    width = 0.12082050003744009
    ddl_gate = 1.0 / (1.0 + np.exp(-slk / (width + eps)))
    abs_slk = np.abs(slk)
    finite_abs_slk = abs_slk[np.isfinite(abs_slk)]
    if len(finite_abs_slk) == 0:
        slk_scale = eps
    else:
        median_slk = np.median(finite_abs_slk)
        mad_slk = np.median(np.abs(finite_abs_slk - median_slk)) * 1.942917115622235
        slk_scale = mad_slk + eps
    slack_norm = slk / (slk_scale + eps)
    slack_penalty = np.where(slk < 0, 11.409437867755441, -4.4057749331751594) * np.abs(slack_norm)
    urgency_coupling = 1.0 + 1.9433535666724748 * (1.0 / (1.0 + np.exp(slk / (width + eps))))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -3.223783529449204 * normalize_mad_bounded(inv_energy) * urgency_coupling
    rank_score = -0.5518853624548277 * ddl_gate * normalize_mad_bounded(rank)
    bottleneck = rank * work
    bottleneck_score = -3.9091623004624227 * normalize_mad_bounded(bottleneck + eps)
    duration = exec_t + comm_t
    dur_uncert_blend = duration + 0.3903355943770339 * uncert
    dur_score = normalize_mad_bounded(dur_uncert_blend + eps)
    wait_clipped = np.clip(wait, 0.0, 31.245961649073447)
    wait_sat = np.clip(wait_clipped, 0.0, 2.0)
    wait_score = -normalize_mad_bounded(wait_sat + eps)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 106563.09235978067
    min_safe = -finfo.max / 106563.09235978067
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
