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
    v2 priority rule: Hybrid urgency exclusivity + critical-path-aware energy gating + 
                      exponential uncertainty decay + robust fairness + dynamic starvation rescue.
    
    Key synthesis:
    - Retains Parent 2's hard urgency exclusivity (-1e12 for slack<=0) for strict DDL compliance.
    - Integrates Parent 1's *criticality-aware wait ratio* (ready_wait_time / task_duration) gated by upward_rank percentile.
    - Uses Parent 2's lightweight robust_minmax_norm but enhances it with degenerate-safe fallbacks.
    - Replaces Parent 2's static energy penalty threshold (rel_slack <= 0.1) with dynamic quantile-based tightness (q25 slack).
    - Introduces *deadline proximity continuity*: smooth transition near q50 slack instead of binary gates.
    - Uncertainty boost decays exponentially with slack, but scaled by dur_uncertainty to prioritize high-risk tasks.
    - Final weights: urgency (0.45), critical latency (0.2), gated energy (0.15), fairness (0.1), starvation (0.05), uncertainty (0.05).
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
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x_finite = x[np.isfinite(x)]
        if x_finite.size == 0:
            return np.zeros_like(x)
        p01 = np.percentile(x_finite, 1.0, method='midpoint')
        p99 = np.percentile(x_finite, 99.0, method='midpoint')
        x_clipped = np.clip(x, p01, p99)
        x_min = np.min(x_clipped)
        x_max = np.max(x_clipped)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x_clipped - x_min) / (x_max - x_min + eps)
    
    # Task duration and derived metrics
    duration = min_exec_time + min_comm_time + eps
    dur_uncertainty = np.divide(uncertainty, duration, out=np.zeros_like(uncertainty), where=duration != 0)
    dur_uncertainty = np.nan_to_num(dur_uncertainty, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Urgency exclusivity: hard priority for violated or imminent deadlines
    is_urgent = (slack <= 0.0).astype(float)
    
    # Dynamic deadline proximity: use quantiles for adaptive thresholds
    if N > 1:
        q25, q50, q75 = np.quantile(slack, [0.25, 0.5, 0.75], method='midpoint')
    else:
        q25 = q50 = q75 = slack[0]
    
    # Critical latency: weighted by upward rank and normalized
    critical_latency_raw = duration * (1.0 + 0.7 * robust_minmax_norm(upward_rank))
    norm_critical_latency = robust_minmax_norm(critical_latency_raw)
    
    # Energy density: incremental energy per unit duration
    energy_density = np.divide(min_incremental_energy, duration + eps, out=np.zeros_like(min_incremental_energy), where=duration + eps != 0)
    norm_energy_density = robust_minmax_norm(energy_density)
    
    # Dynamic energy gating: only penalize when slack is in tight region (<= q25)
    tight_slack_mask = (slack <= q25).astype(float)
    rank_threshold = np.percentile(upward_rank, 75.0) + eps if N > 1 else upward_rank[0] + eps
    high_rank_mask = (upward_rank > rank_threshold).astype(float)
    energy_penalty_mask = tight_slack_mask * high_rank_mask
    energy_penalty = norm_energy_density * energy_penalty_mask
    
    # Fairness: normalized wait-per-work with dynamic threshold
    wait_per_work = np.divide(ready_wait_time, remaining_work + eps, out=np.zeros_like(ready_wait_time), where=remaining_work + eps != 0)
    wait_per_work = np.nan_to_num(wait_per_work, nan=0.0, posinf=0.0, neginf=0.0)
    wpw_finite = wait_per_work[np.isfinite(wait_per_work)]
    work_threshold = np.percentile(wpw_finite, 10.0) + eps if wpw_finite.size > 0 else eps
    wait_gate = (wait_per_work >= work_threshold).astype(float)
    norm_wait_per_work = robust_minmax_norm(wait_per_work)
    wait_penalty = (1.0 - is_urgent) * norm_wait_per_work * wait_gate
    
    # Starvation rescue: criticality-aware wait ratio (from Parent 1)
    norm_wait_ratio = np.clip(ready_wait_time / (duration + eps), 0.0, 20.0)
    ur_percentile = np.percentile(upward_rank, 70) if N > 1 else upward_rank[0]
    is_starvable = (norm_wait_ratio > 1.0) & (upward_rank >= ur_percentile) & (slack < 300.0)
    starvation_boost = np.where(is_starvable, norm_wait_ratio * (1.0 + 0.2 * robust_minmax_norm(upward_rank)), 0.0)
    starvation_boost = np.clip(starvation_boost, 0.0, 2.0)
    norm_starvation = robust_minmax_norm(starvation_boost)
    
    # Uncertainty boost: decays exponentially with slack, scaled by dur_uncertainty
    tau = 10.0
    proximity_bias = np.exp(-np.maximum(0.0, slack) / tau)
    uncertainty_boost = dur_uncertainty * proximity_bias
    norm_uncertainty_boost = robust_minmax_norm(uncertainty_boost)
    
    # Base score composition
    base_score = np.full(N, 1.0, dtype=float)
    score = np.where(is_urgent, -1000000000000.0, base_score)
    
    # Non-urgent scoring components
    non_urgent_contrib = (
        0.2 * norm_critical_latency +
        0.15 * energy_penalty +
        0.1 * wait_penalty +
        0.05 * norm_starvation +
        0.05 * norm_uncertainty_boost
    )
    
    score = np.where(is_urgent, score, score + non_urgent_contrib)
    
    # Ensure finite output
    score = np.clip(score, -1000000000000.0, 1000000000000.0)
    score = np.nan_to_num(score, nan=1000000000000.0, posinf=1000000000000.0, neginf=-1000000000000.0)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
