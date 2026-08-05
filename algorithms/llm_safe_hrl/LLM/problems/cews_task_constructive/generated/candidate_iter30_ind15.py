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
    v2: Refined lexicographic priority with stabilized urgency, adaptive fairness,
    post-normalization violation penalty, and slack-gated energy-aware criticality.
    
    Key improvements:
    - Stabilized arctan urgency using larger scale factor (0.2) to reduce noise near zero slack
    - Restored log-scaled fairness (log1p) for stronger starvation mitigation under uncertainty
    - Moved hard violation penalty *after* quantile normalization to preserve intra-feasible ordering
    - Introduced slack-adaptive critical efficiency weight: exp(-max(0, -robust_slack)/tau) to smoothly suppress energy optimization when deadline risk rises
    - Added latency pressure normalization by upward_rank to emphasize work density per criticality unit
    - All terms bounded, sanitized, and quantile-normalized independently before weighted combination
    """
    eps = 1e-08
    
    def sanitize(x):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(x, nan=eps, posinf=1e6, neginf=-1e6)
    
    min_exec_time = sanitize(min_exec_time)
    min_comm_time = sanitize(min_comm_time)
    min_incremental_energy = sanitize(min_incremental_energy)
    slack = sanitize(slack)
    upward_rank = sanitize(upward_rank)
    remaining_work = sanitize(remaining_work)
    ready_wait_time = sanitize(ready_wait_time)
    uncertainty = sanitize(uncertainty)
    
    # Robust slack: risk-adjusted margin (positive = safe, negative = violated/risky)
    robust_slack = np.where(slack > 0, slack - 1.5 * uncertainty, slack)
    
    # Urgency: smooth, bounded [0,1] arctan scaling with reduced sensitivity near zero
    # Larger denominator (0.2) prevents steep gradients → less noise in tight-deadline regimes
    urgency_raw = (np.arctan(-robust_slack / (0.2 + eps)) + np.pi / 2) / np.pi
    
    # Latency pressure: critical-path work density per time unit, normalized by rank to avoid rank-dominance
    exec_comm_sum = min_exec_time + min_comm_time + eps
    latency_pressure = (remaining_work + eps) / (exec_comm_sum + eps)
    latency_pressure = np.where(upward_rank > eps, latency_pressure * upward_rank, 0.0)
    
    # Critical efficiency: energy-efficient critical work — gated AND scaled by deadline safety
    critical_efficiency = (upward_rank + eps) * (remaining_work + eps) / (min_incremental_energy + eps)
    # Soft gating: exponentially decay weight as robust_slack becomes negative
    tau = np.clip(np.abs(robust_slack) + 0.1, 0.1, 10.0)
    safety_weight = np.exp(-np.clip(-robust_slack, 0.0, 10.0) / tau)
    critical_efficiency = np.where(robust_slack > -eps, critical_efficiency * safety_weight, 0.0)
    
    # Energy term: marginal energy per time unit, only active when slack is positive
    energy_term = min_incremental_energy / (exec_comm_sum + eps)
    energy_term = energy_term * np.clip(robust_slack / (1.0 + eps), 0.0, 1.0)
    
    # Fairness: log-scaled wait time amplified by uncertainty (stronger starvation prevention)
    fairness_base = np.log1p(ready_wait_time + eps) * (1.0 + 0.6 * np.clip(uncertainty, 0.0, 1.0))
    
    def robust_quantile_normalize(x):
        x = np.clip(x, -1e6, 1e6)
        if x.size == 1:
            return np.array([0.0])
        q1, q3 = np.quantile(x, [0.25, 0.75])
        iqr = q3 - q1
        if iqr < eps:
            return np.zeros_like(x)
        return (x - q1) / (iqr + eps)
    
    norm_urgency = robust_quantile_normalize(urgency_raw)
    norm_latency = robust_quantile_normalize(latency_pressure)
    norm_crit_eff = robust_quantile_normalize(critical_efficiency)
    norm_energy = robust_quantile_normalize(energy_term)
    norm_fair = robust_quantile_normalize(fairness_base)
    
    # Weighted sum: urgency dominates, then latency/critical-efficiency, then energy, fairness last
    score = (
        -28.0 * norm_urgency 
        - 14.0 * norm_latency 
        - 10.0 * norm_crit_eff 
        - 4.0 * norm_energy 
        + 0.35 * norm_fair
    )
    
    # Apply hard violation penalty *after* normalization to preserve relative order among feasible tasks
    violation_mask = robust_slack < -eps
    if np.any(violation_mask):
        base_ref = np.min(score[~violation_mask]) if np.any(~violation_mask) else np.min(score)
        score = np.where(violation_mask, base_ref - 1e9, score)
    
    # Final sanitization and clipping
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)
    
    return score.reshape(-1)
