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
    v2: Hybrid lexicographic priority with arctan urgency, quantile normalization,
    slack-gated critical efficiency, and starvation-aware fairness.
    Combines Parent 2's robust deadline pressure and outlier resilience with Parent 1's
    explicit latency-pressure modeling and smoother fairness amplification.
    Key improvements:
    - Uses robust_slack = slack - 1.5*uncertainty for risk-aware deadline margin
    - Arctan urgency scaled to [0,1] with monotonic, bounded sensitivity near DDL
    - Critical efficiency gated by robust_slack > 0 AND upward_rank > eps (avoids noise)
    - Latency pressure computed as (upward_rank * remaining_work) / (min_exec_time + min_comm_time + eps)
      — captures critical-path work density per time unit
    - Fairness term uses sqrt-scaled wait time * (1 + uncertainty) for gentle but effective starvation prevention
    - Hard violation penalty applied *before* normalization to preserve lexicographic order
    - All divisions guarded; all arrays sanitized and clipped to finite bounds
    """
    eps = 1e-08
    
    # Sanitize inputs: ensure finite, replace NaN/inf with safe values
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
    
    # Risk-adjusted slack: subtract uncertainty budget to reflect worst-case margin
    robust_slack = np.where(slack > 0, slack - 1.5 * uncertainty, slack)
    
    # Hard deadline violation mask: tasks already predicted to miss deadline
    violation_mask = robust_slack < -eps
    
    # Arctan-based urgency: smooth, bounded [0,1], high sensitivity near zero robust_slack
    # arctan(x) ∈ (-π/2, π/2) → (arctan(-robust_slack) + π/2) / π ∈ (0,1)
    urgency_raw = (np.arctan(-robust_slack / (0.05 + eps)) + np.pi/2) / np.pi
    
    # Latency pressure: critical-path work density per time unit (higher = more urgent bottleneck)
    exec_comm_sum = min_exec_time + min_comm_time + eps
    latency_pressure = (upward_rank + eps) * (remaining_work + eps) / (exec_comm_sum + eps)
    
    # Critical efficiency: prioritize energy-efficient execution *only* when slack is positive
    # Avoids wasting energy on doomed tasks; requires meaningful upward_rank to avoid noise
    critical_efficiency = (upward_rank + eps) * (remaining_work + eps) / (min_incremental_energy + eps)
    critical_efficiency = np.where((robust_slack > 0) & (upward_rank > eps), critical_efficiency, 0.0)
    
    # Energy term: marginal energy per time unit, down-weighted when slack is tight
    energy_term = min_incremental_energy / (exec_comm_sum + eps)
    energy_weight = np.clip(robust_slack / (1.0 + eps), 0.0, 1.0)  # 0 when slack ≤ 0
    energy_term = energy_term * energy_weight
    
    # Fairness: sqrt-scaled wait time amplified by uncertainty (gentler than log, avoids singularity)
    # Prevents starvation without dominating deadline-critical decisions
    fairness_base = np.sqrt(ready_wait_time + eps) * (1.0 + 0.4 * np.clip(uncertainty, 0.0, 1.0))
    
    # Quantile-based normalization: robust to outliers (Q1/Q3 instead of min/max)
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
    
    # Lexicographic score: urgency dominates, then latency, critical efficiency, energy, fairness last
    score = (
        -32.0 * norm_urgency 
        - 16.0 * norm_latency 
        - 12.0 * norm_crit_eff 
        - 6.0 * norm_energy 
        + 0.25 * norm_fair
    )
    
    # Apply hard penalty for violations: push below all non-violating scores
    if np.any(violation_mask):
        non_viol_scores = score[~violation_mask]
        base_ref = np.min(non_viol_scores) if len(non_viol_scores) > 0 else np.min(score)
        score = np.where(violation_mask, base_ref - 1e9, score)
    
    # Final sanitization: ensure finite output
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)
    
    # Enforce shape (N,) — no scalars, no column vectors
    return score.reshape(-1)
