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
    v2 priority rule: Absolute urgency dominance + critical-path-aware risk gating + robust fairness amplification.
    
    Key innovations:
    - Hard urgency enforcement via -1e12 offset (guarantees slack<=0 tasks always win)
    - Critical-latency penalty scaled by upward_rank * normalized duration, not raw duration
    - Energy penalty uses rank-normalized energy density with 95th-percentile thresholding
    - Uncertainty coupling gated by both |slack| AND upward_rank (high-risk + high-criticality only)
    - Fairness applied uniformly to non-urgent tasks via robust-minmax wait time scaling
    - All normalizations use 1%-99% clipping + safe min-max; all divisions guarded; NaN/inf sanitized
    - Final weights hierarchical: urgency (dominant) > critical-latency (0.25) > energy (0.22) 
      > fairness (0.12) > uncertainty (0.09) > remaining_work (0.04)
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
        p01 = np.percentile(x, 1.0)
        p99 = np.percentile(x, 99.0)
        x_clipped = np.clip(x, p01, p99)
        x_min = np.min(x_clipped)
        x_max = np.max(x_clipped)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x_clipped - x_min) / (x_max - x_min + eps)

    # Absolute urgency: slack <= 0 tasks get lowest possible score
    is_urgent = (slack <= 0.0).astype(float)
    
    # Task duration for normalization and scaling
    duration = min_exec_time + min_comm_time + eps
    
    # Critical latency: duration weighted by upward rank importance, then normalized
    critical_latency_raw = duration * (1.0 + 0.8 * robust_minmax_norm(upward_rank))
    norm_critical_latency = robust_minmax_norm(critical_latency_raw)
    
    # Energy penalty: energy density normalized by duration, then rank-weighted and thresholded
    energy_density = min_incremental_energy / (duration + eps)
    rank_weighted_energy = energy_density / (upward_rank + eps)
    threshold_rank_energy = np.percentile(rank_weighted_energy, 95.0) + eps
    energy_penalty_mask = (rank_weighted_energy > threshold_rank_energy).astype(float)
    norm_energy_density = robust_minmax_norm(energy_density)
    energy_penalty = norm_energy_density * energy_penalty_mask
    
    # Fairness: wait time prioritization only for non-urgent tasks
    norm_wait_time = robust_minmax_norm(ready_wait_time)
    wait_penalty = (1.0 - is_urgent) * norm_wait_time
    
    # Uncertainty boost: activated only for high-criticality (upward_rank) AND high-risk (small |slack|)
    slack_abs = np.abs(slack) + 1.0
    slack_scale_factor = np.clip(1.0 / slack_abs, 0.1, 10.0)
    rank_gate = robust_minmax_norm(upward_rank)
    uncertainty_boost = uncertainty * slack_scale_factor * rank_gate
    norm_uncertainty_boost = robust_minmax_norm(uncertainty_boost)
    
    # Remaining work term for load balancing in long-tail workflows
    norm_remaining_work = robust_minmax_norm(remaining_work)
    
    # Base score starts at 0.0; urgent tasks get massive negative offset
    base_score = np.where(is_urgent, -1e12, 0.0)
    
    # Hierarchical weighted composition
    score = (
        base_score +
        0.25 * norm_critical_latency +
        0.22 * energy_penalty +
        0.12 * wait_penalty +
        0.09 * norm_uncertainty_boost +
        0.04 * norm_remaining_work
    )
    
    # Sanitize numerical artifacts
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    
    # Ensure shape matches input
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
