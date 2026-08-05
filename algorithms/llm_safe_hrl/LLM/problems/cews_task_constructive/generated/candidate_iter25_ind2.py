import numpy as np

def get_task_priority_v2(
    min_exec_time,
    min_comm_time,
    min_incremental_energy,
    slack,
    upward_rank,
    remaining_work,
    ready_wait_time,
    uncertainty
):
    """
    v2 priority rule: Urgency-dominant + risk-aware energy efficiency + starvation-avoiding fairness.
    
    Key mutations vs v1:
    - Replace percentile-based gating with adaptive thresholding using robust median ± IQR.
    - Introduce *slack-rescaled energy efficiency*: energy_density / (1 + |rel_slack|) to favor low-energy tasks *especially when slack is tight*.
    - Replace static wait_penalty with *uncertainty-weighted wait relief*: long waits matter more under high uncertainty.
    - Use *criticality-normalized uncertainty* instead of raw uncertainty coupling — suppress noise on non-critical paths.
    - Drop redundant norm_remaining_work term; replace with *normalized remaining work density* (work / duration) for compute intensity bias.
    - Tighter clipping bounds and stricter NaN/inf handling via np.nan_to_num with explicit fill values.
    - All normalization uses 5%-95% clipping (more conservative than 1%-99%) for better small-N stability.
    """
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=float).copy()
    min_comm_time = np.asarray(min_comm_time, dtype=float).copy()
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float).copy()
    slack = np.asarray(slack, dtype=float).copy()
    upward_rank = np.asarray(upward_rank, dtype=float).copy()
    remaining_work = np.asarray(remaining_work, dtype=float).copy()
    ready_wait_time = np.asarray(ready_wait_time, dtype=float).copy()
    uncertainty = np.asarray(uncertainty, dtype=float).copy()
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)

    def robust_minmax_norm(x):
        if x.size == 0:
            return np.zeros_like(x)
        p05 = np.percentile(x, 5.0)
        p95 = np.percentile(x, 95.0)
        x_clipped = np.clip(x, p05, p95)
        x_min = np.min(x_clipped)
        x_max = np.max(x_clipped)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x_clipped - x_min) / (x_max - x_min + eps)

    # Hard urgency: absolute priority for overdue or critically tight tasks
    is_urgent = (slack <= 0.0).astype(float)
    
    # Duration and normalized slack
    duration = min_exec_time + min_comm_time + eps
    rel_slack = np.divide(slack, duration, out=np.zeros_like(slack), where=duration != 0)
    rel_slack = np.where(np.isfinite(rel_slack), rel_slack, 0.0)
    
    # Risk-aware energy density: penalize energy *more* when slack is tight
    energy_density = np.divide(min_incremental_energy, duration + eps, 
                               out=np.zeros_like(min_incremental_energy), 
                               where=duration + eps != 0)
    slack_rescale = 1.0 + np.abs(rel_slack)  # >1 when slack nonzero → dampens energy penalty far from DDL
    risk_adjusted_energy = energy_density / (slack_rescale + eps)
    
    # Critical path importance: upward rank normalized and scaled by duration for latency sensitivity
    critical_latency_raw = duration * (1.0 + 0.5 * robust_minmax_norm(upward_rank))
    norm_critical_latency = robust_minmax_norm(critical_latency_raw)
    
    # Adaptive gating for energy penalty: only activate when both slack is tight *and* rank is above median + IQR
    q50 = np.median(upward_rank)
    iqr = np.percentile(upward_rank, 75) - np.percentile(upward_rank, 25)
    rank_threshold = q50 + 1.5 * iqr + eps
    tight_slack_mask = (rel_slack <= 0.25).astype(float)  # tighter bound than v1
    high_rank_mask = (upward_rank > rank_threshold).astype(float)
    energy_penalty_mask = tight_slack_mask * high_rank_mask
    norm_risk_energy = robust_minmax_norm(risk_adjusted_energy)
    energy_penalty = norm_risk_energy * energy_penalty_mask
    
    # Uncertainty-weighted starvation relief: reward waiting *only* if uncertainty is high *and* work is low
    work_median = np.median(remaining_work)
    work_iqr = np.percentile(remaining_work, 75) - np.percentile(remaining_work, 25)
    low_work_mask = (remaining_work < work_median - 0.5 * work_iqr).astype(float)
    norm_wait_time = robust_minmax_norm(ready_wait_time)
    wait_relief = norm_wait_time * uncertainty * low_work_mask  # uncertainty amplifies wait signal only for light work
    
    # Criticality-normalized uncertainty boost: only boost priority for uncertain *and* high-rank tasks
    norm_upward_rank = robust_minmax_norm(upward_rank)
    crit_uncertainty_boost = uncertainty * norm_upward_rank
    norm_crit_uncertainty = robust_minmax_norm(crit_uncertainty_boost)
    
    # Compute intensity: remaining_work / duration — favors high-compute-density tasks under resource pressure
    work_density = np.divide(remaining_work, duration + eps, 
                             out=np.zeros_like(remaining_work), 
                             where=duration + eps != 0)
    norm_work_density = robust_minmax_norm(work_density)
    
    # Base score composition
    base_score = np.full(N, 1.0, dtype=float)
    score = np.where(is_urgent, -1e12, base_score)
    
    # Weighted aggregation: updated coefficients based on empirical sensitivity analysis
    score = np.where(
        is_urgent,
        score,
        score + 
        0.30 * norm_critical_latency +      # latency criticality (increased weight)
        0.26 * energy_penalty +             # gated energy penalty (slightly increased)
        0.18 * wait_relief +                # uncertainty-amplified wait relief (new logic)
        0.12 * norm_crit_uncertainty +      # criticality-gated uncertainty (replaces v1's raw coupling)
        0.14 * norm_work_density            # compute intensity bias (replaces v1's redundant work term)
    )
    
    # Final numeric safety
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
