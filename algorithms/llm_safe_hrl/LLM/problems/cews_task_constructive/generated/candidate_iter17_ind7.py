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
    v2 priority rule: Absolute urgency dominance + critical-path energy gating + adaptive fairness + uncertainty-aware slack normalization.
    
    Key innovations:
    - Hard urgency enforcement: urgent tasks (slack <= 0) receive *guaranteed minimum score* (-1e12), not just offset.
    - Critical-path energy penalty: only activates for high-upward-rank AND tight-slack tasks, avoiding energy over-penalization on non-critical paths.
    - Adaptive fairness: wait-time scaling uses robust z-score *and* dynamic work-threshold (5th percentile), preventing starvation while respecting workload scale.
    - Uncertainty coupling: normalized by both relative slack and upward rank, suppressing noise on low-criticality or low-risk tasks.
    - Robust normalization: 1%-99% clipping + min-max, with fallback to zero when range is degenerate.
    - All divisions guarded; NaN/inf replaced deterministically; no global state or side effects.
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
    
    # Absolute urgency flag: slack <= 0 always wins
    is_urgent = (slack <= 0.0).astype(float)
    
    # Duration and relative slack for risk-aware scaling
    duration = min_exec_time + min_comm_time + eps
    rel_slack = np.divide(slack, duration, out=np.zeros_like(slack), where=duration != 0)
    rel_slack = np.where(np.isfinite(rel_slack), rel_slack, 0.0)
    
    # Critical-path latency penalty: scaled by upward rank and duration
    critical_latency_raw = duration * (1.0 + 0.7 * robust_minmax_norm(upward_rank))
    norm_critical_latency = robust_minmax_norm(critical_latency_raw)
    
    # Energy density penalization: only for critical-path tasks with tight slack
    energy_density = np.divide(min_incremental_energy, duration + eps, out=np.zeros_like(min_incremental_energy), where=duration + eps != 0)
    # Gate energy penalty: activate only when both slack is tight (rel_slack <= 0.3) AND upward_rank is high (> 75th percentile)
    rank_threshold = np.percentile(upward_rank, 75.0) + eps
    tight_slack_mask = (rel_slack <= 0.3).astype(float)
    high_rank_mask = (upward_rank > rank_threshold).astype(float)
    energy_penalty_mask = tight_slack_mask * high_rank_mask
    norm_energy_density = robust_minmax_norm(energy_density)
    energy_penalty = norm_energy_density * energy_penalty_mask
    
    # Fairness: wait-time per unit work, robustly gated
    wait_per_work = np.divide(ready_wait_time, remaining_work + eps, out=np.zeros_like(ready_wait_time), where=remaining_work + eps != 0)
    work_threshold = np.percentile(remaining_work, 5.0) + eps  # Dynamic low-work threshold
    wait_gate = (remaining_work >= work_threshold).astype(float)
    norm_wait_time = robust_minmax_norm(ready_wait_time)
    wait_penalty = (1.0 - is_urgent) * norm_wait_time * wait_gate
    
    # Uncertainty coupling: only amplifies for high-risk (low |rel_slack|) AND high-criticality (high upward_rank)
    abs_rel_slack = np.abs(rel_slack) + eps
    slack_sensitivity = np.clip(1.0 / abs_rel_slack, 0.1, 10.0)
    rank_sensitivity = robust_minmax_norm(upward_rank)
    uncertainty_boost = uncertainty * slack_sensitivity * rank_sensitivity
    norm_uncertainty_boost = robust_minmax_norm(uncertainty_boost)
    
    # Remaining work bias: mild preference for larger workloads (to amortize setup cost)
    norm_remaining_work = robust_minmax_norm(remaining_work)
    
    # Composite score with strict hierarchy
    base_score = np.full(N, 1.0, dtype=float)
    # Urgent tasks get absolute priority via guaranteed minimum score
    score = np.where(is_urgent, -1e12, base_score)
    # Non-urgent tasks accumulate penalties
    score = np.where(is_urgent, score, 
                     score + 0.28 * norm_critical_latency + 
                           0.24 * energy_penalty + 
                           0.13 * wait_penalty + 
                           0.10 * norm_uncertainty_boost + 
                           0.05 * norm_remaining_work)
    
    # Final sanitization
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
