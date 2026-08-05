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
    v2: Hybrid urgency-energy-fairness priority with robust uncertainty-aware normalization.
    
    Key innovations:
    - Unified robust_slack = slack - 2*uncertainty, clipped and MAD-normalized for stability
    - Urgency: smooth exp(-robust_slack / scale) with adaptive scale using IQR-based dispersion
    - Criticality: upward_rank * (remaining_work / exec_effort) gated by slack-confidence sigmoid
    - Energy efficiency: min_incremental_energy / (remaining_work + eps) activated only when robust_slack > 0
    - Fairness: exponential aging scaled by urgency pressure (1 + max(0,-robust_slack)) + wait_ratio clipping
    - Uncertainty integration: multiplicative penalty on latency-sensitive tasks (0 <= robust_slack <= 3*Q3_robust)
    - All components normalized via robust_mad_normalize (IQR fallback → MAD → constant zero for degenerate cases)
    - Final score bounded, finite, deterministic, and satisfies all interface contracts.
    """
    eps = 1e-08
    N = len(slack)
    if N == 0:
        return np.array([], dtype=float)
    
    # Sanitize inputs: convert, replace NaN/inf with safe values
    def sanitize(x):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(x, nan=eps, posinf=1e6, neginf=eps)
    
    min_exec_time = sanitize(min_exec_time)
    min_comm_time = sanitize(min_comm_time)
    min_incremental_energy = sanitize(min_incremental_energy)
    slack = sanitize(slack)
    upward_rank = sanitize(upward_rank)
    remaining_work = sanitize(remaining_work)
    ready_wait_time = sanitize(ready_wait_time)
    uncertainty = sanitize(uncertainty)
    
    # Robust slack: account for uncertainty margin; clip extremes
    robust_slack = slack - 2.0 * uncertainty
    robust_slack = np.clip(robust_slack, -1e6, 1e6)
    
    # Adaptive scale for urgency: use IQR of |robust_slack| to avoid outlier sensitivity
    abs_robust = np.abs(robust_slack)
    q25, q75 = np.percentile(abs_robust, [25, 75]) if N > 2 else (eps, eps)
    scale = q75 - q25 if (q75 - q25) > eps else np.median(abs_robust) + eps
    
    # Urgency term: smooth, monotonic, higher penalty for negative robust_slack
    urgency_raw = np.exp(-robust_slack / (scale + eps))
    urgency_raw = np.clip(urgency_raw, 0.0, 1e6)
    
    # Critical path density: rank × work / effort, confidence-gated by robust_slack
    exec_effort = np.maximum(min_exec_time, eps)
    base_critical = upward_rank * (remaining_work / (exec_effort + eps))
    # Confidence gate: sigmoid on robust_slack → 0.5 at slack=0, near 1 when slack large
    conf_gate = 1.0 / (1.0 + np.exp(-robust_slack / (scale + eps)))
    critical_density = base_critical * conf_gate
    
    # Energy efficiency term: only active when robust_slack > 0 (safe to optimize)
    energy_per_work = min_incremental_energy / (remaining_work + eps)
    energy_gate = np.where(robust_slack > 0, 
                          0.1 + 0.9 / (1.0 + np.exp(-(robust_slack - 0.5) / 0.8)), 
                          0.0)
    efficiency_score = energy_per_work * energy_gate
    
    # Fairness: aging boost amplified under lateness risk + bounded wait ratio
    aging_factor = 1.0 + np.maximum(0.0, -robust_slack)
    exp_aging = 1.0 - np.exp(-np.clip(ready_wait_time, 0.0, 20.0 * (np.median(exec_effort) + eps)) 
                            / (np.median(exec_effort) + eps))
    wait_ratio = np.clip(ready_wait_time / (min_exec_time + min_comm_time + eps), 0.0, 10.0)
    fairness_boost = wait_ratio + 0.4 * exp_aging * aging_factor
    
    # Uncertainty-latency penalty: only applied in "tight but not violated" window
    pos_robust = robust_slack[robust_slack > 0]
    median_pos = np.median(pos_robust) if len(pos_robust) > 0 else 1.0
    tight_mask = (robust_slack >= 0) & (robust_slack <= 3.0 * median_pos)
    unc_penalty = np.where(tight_mask, 
                          uncertainty * (min_exec_time + min_comm_time), 
                          0.0)
    
    # Robust MAD-based normalization function handling N=1, constant arrays, outliers
    def robust_mad_normalize(x):
        if N == 1:
            return np.zeros(1, dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        median_x = np.median(x)
        abs_dev = np.abs(x - median_x)
        mad = np.median(abs_dev)
        if mad < eps:
            # All values near median → return zeros
            return np.zeros_like(x)
        normed = (x - median_x) / (mad + eps)
        return np.clip(normed, -5.0, 5.0)
    
    # Normalize each component
    urgency_norm = robust_mad_normalize(urgency_raw)
    critical_norm = robust_mad_normalize(critical_density)
    efficiency_norm = robust_mad_normalize(efficiency_score)
    fairness_norm = robust_mad_normalize(fairness_boost)
    latency_norm = robust_mad_normalize(unc_penalty)
    
    # Weighted linear combination: prioritize urgency & criticality, moderate fairness & latency, suppress energy when urgent
    score = (
        +4.2 * urgency_norm           # Strongest weight: deadline adherence is primary
        - 2.8 * critical_norm        # High weight: critical path drives feasibility
        - 0.75 * efficiency_norm     # Modest weight: energy only optimized when safe
        + 0.25 * fairness_norm       # Light boost: prevent starvation without compromising DDL
        + 0.5 * latency_norm         # Penalty for uncertain latency in tight windows
    )
    
    # Final safeguard: clamp and sanitize
    score = np.clip(score, -1e10, 1e10)
    score = np.nan_to_num(score, nan=1e10, posinf=1e10, neginf=-1e10)
    
    return score
