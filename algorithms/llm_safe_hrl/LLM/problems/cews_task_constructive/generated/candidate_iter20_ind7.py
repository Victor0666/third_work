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
    v2 priority rule: Hybrid urgency dominance + critical-path energy gating + 
                      smooth deadline proximity + adaptive fairness + uncertainty-gated robustness.
    
    Key synthesis:
    - Hard urgency enforcement (Parent 2): urgent tasks (slack <= 0) get guaranteed min score (-1e12)
    - Smooth slack sensitivity near deadline (Parent 1): sigmoid(-5*slack) for graceful prioritization when slack ≈ 0
    - Critical-path energy gating (Parent 2): only penalize energy on high-rank AND tight-slack tasks
    - Adaptive fairness with wait-per-work and dynamic threshold (Parent 2), but normalized via robust minmax (Parent 1)
    - Uncertainty boost gated by both relative slack *and* upward rank (Parent 2) + smoothed via sigmoid (Parent 1 style)
    - All normalizations use percentile-robust minmax with degenerate-range fallback
    - Final weights rebalanced: urgency dominates (0.4), critical latency (0.22), gated energy (0.18), 
      adaptive fairness (0.1), uncertainty boost (0.07), remaining work (0.03)
    """
    eps = 1e-08
    # Ensure float dtype and safe copying
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
        p01 = np.percentile(x, 1.0, method='midpoint')
        p99 = np.percentile(x, 99.0, method='midpoint')
        x_clipped = np.clip(x, p01, p99)
        x_min = np.min(x_clipped)
        x_max = np.max(x_clipped)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x_clipped - x_min) / (x_max - x_min + eps)
    
    # Hard urgency enforcement: immediate scheduling for violated deadlines
    is_urgent = (slack <= 0.0).astype(float)
    
    # Duration and relative slack for robust deadline proximity
    duration = min_exec_time + min_comm_time + eps
    rel_slack = np.divide(slack, duration, out=np.zeros_like(slack), where=duration != 0)
    rel_slack = np.where(np.isfinite(rel_slack), rel_slack, 0.0)
    
    # Smooth urgency bias near deadline (sigmoid-based, from Parent 1)
    # Rises from ~0 (far ahead) to ~1 (violated), preserving exploration at slack≈0
    smooth_urgency_bias = 1.0 / (1.0 + np.exp(-5.0 * slack))
    
    # Critical latency: duration weighted by upward rank importance
    critical_latency_raw = duration * (1.0 + 0.7 * robust_minmax_norm(upward_rank))
    norm_critical_latency = robust_minmax_norm(critical_latency_raw)
    
    # Energy density and critical-path gating (Parent 2 style)
    energy_density = np.divide(min_incremental_energy, duration + eps, 
                              out=np.zeros_like(min_incremental_energy), where=duration + eps != 0)
    rank_threshold = np.percentile(upward_rank, 75.0) + eps
    tight_slack_mask = (rel_slack <= 0.3).astype(float)
    high_rank_mask = (upward_rank > rank_threshold).astype(float)
    energy_penalty_mask = tight_slack_mask * high_rank_mask
    norm_energy_density = robust_minmax_norm(energy_density)
    energy_penalty = norm_energy_density * energy_penalty_mask
    
    # Adaptive fairness: wait-time per work, thresholded by low-work tasks
    wait_per_work = np.divide(ready_wait_time, remaining_work + eps, 
                             out=np.zeros_like(ready_wait_time), where=remaining_work + eps != 0)
    work_threshold = np.percentile(remaining_work, 5.0) + eps
    wait_gate = (remaining_work >= work_threshold).astype(float)
    norm_wait_time = robust_minmax_norm(ready_wait_time)
    wait_penalty = (1.0 - is_urgent) * norm_wait_time * wait_gate
    
    # Uncertainty boost: gated by both slack proximity and criticality, smoothed
    abs_rel_slack = np.abs(rel_slack) + eps
    slack_sensitivity = np.clip(1.0 / abs_rel_slack, 0.1, 10.0)  # bounded sensitivity
    rank_sensitivity = robust_minmax_norm(upward_rank)
    uncertainty_boost = uncertainty * slack_sensitivity * rank_sensitivity
    norm_uncertainty_boost = robust_minmax_norm(uncertainty_boost)
    
    # Remaining work normalization
    norm_remaining_work = robust_minmax_norm(remaining_work)
    
    # Final score: hard urgency dominates; others contribute only for non-urgent tasks
    base_score = np.full(N, 1.0, dtype=float)
    score = np.where(is_urgent, -1000000000000.0, base_score)
    score = np.where(
        is_urgent,
        score,
        score + 
        0.22 * norm_critical_latency + 
        0.18 * energy_penalty + 
        0.1 * wait_penalty + 
        0.07 * norm_uncertainty_boost + 
        0.03 * norm_remaining_work
    )
    
    # Apply smooth urgency bias as primary term (replaces linear weight for urgency)
    # Ensures monotonic prioritization near deadline without hard cliffs except violation
    score = score * (1.0 - smooth_urgency_bias) + (-1000000000000.0) * smooth_urgency_bias
    
    # Clip and sanitize
    score = np.clip(score, -1000000000000.0, 1000000000000.0)
    score = np.nan_to_num(score, nan=1000000000000.0, posinf=1000000000000.0, neginf=-1000000000000.0)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
