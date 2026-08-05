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
    Hybrid priority rule combining v0's robust harmonic efficiency and urgency ramp
    with v1's relative slack, adaptive energy scaling, and context-aware starvation control.
    
    Key innovations:
    - Uses *relative slack* (slack/duration) for cross-task urgency calibration,
      but applies smooth sigmoid ramp (v0) instead of piecewise linear (v1) for stability.
    - Harmonic energy-work efficiency: 2 / (1/(energy_per_work+eps) + 1/(work_efficiency_ratio+eps))
      replaces simple energy normalization — more robust to outliers and zero-work cases.
    - Adaptive energy risk weighting: min_incremental_energy * (1+uncertainty)^alpha,
      where alpha = clip(1.0 + 0.65 * max(0,-slack), 1.0, 4.0) — stronger amplification than v1.
    - Contextual energy scaling: only applied when upward_rank > median AND remaining_work > p25,
      preventing over-penalization of low-criticality tasks.
    - Starvation penalty: adaptive, bounded [0, 0.3], scaled by (ready_wait_time / p95_wait) * 
      min(1.0, max(0, -slack)/max(eps, median_duration)) * (remaining_work / median_rw),
      ensuring fairness only for genuinely starved, high-work, at-risk tasks.
    - Unified robust normalization using median-IQR with tight clipping [-8,8] and fallback variance handling.
    - All operations eps-protected, nan/inf guarded, deterministic, and shape-compliant.
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
    
    def robust_normalize(x):
        if x.size == 0:
            return np.zeros_like(x)
        med = np.median(x)
        q1, q3 = np.quantile(x, [0.25, 0.75], method='midpoint')
        iqr = q3 - q1 + eps
        iqr = np.where(iqr < eps, np.std(x, ddof=0) + eps, iqr)
        norm = (x - med) / iqr
        return np.clip(norm, -8.0, 8.0)
    
    task_min_duration = np.maximum(min_exec_time + min_comm_time, eps)
    rel_slack = slack / task_min_duration
    # Smooth urgency ramp via sigmoid on relative slack (v0-style, but with v1's relative base)
    deadline_urgency = 1.0 / (1.0 + np.exp(-rel_slack * 2.0))
    
    # Stronger risk exponent: v0's 0.8 → v1's 0.5 → hybrid 0.65, capped [1.0, 4.0]
    risk_exponent = np.clip(1.0 + 0.65 * np.maximum(0.0, -slack), 1.0, 4.0)
    energy_risk_weighted = min_incremental_energy * np.power(1.0 + uncertainty, risk_exponent)
    energy_safe = np.maximum(energy_risk_weighted, eps)
    
    # Harmonic energy-work efficiency (v0's core strength)
    energy_per_work = min_incremental_energy / (remaining_work + eps)
    work_efficiency_ratio = remaining_work / (min_incremental_energy + eps)
    harmonic_eff = 2.0 / (1.0 / (energy_per_work + eps) + 1.0 / (work_efficiency_ratio + eps))
    harmonic_eff = np.clip(harmonic_eff, 1e-06, 1e6)
    norm_energy_eff = robust_normalize(harmonic_eff)
    
    # Criticality-energy ratio: upward_rank / energy_safe, normalized
    crit_eff_ratio = upward_rank / energy_safe
    crit_eff_ratio = np.clip(crit_eff_ratio, 1e-06, 1e6)
    crit_eff_norm = robust_normalize(crit_eff_ratio)
    
    # Contextual energy scaling: only for high-criticality & sufficient work (v1-inspired guard)
    median_ur = np.median(upward_rank) + eps
    median_rw = np.median(remaining_work) + eps
    p25_rw = np.quantile(remaining_work, 0.25, method='midpoint') + eps
    energy_work_weight = np.where(
        (upward_rank > median_ur) & (remaining_work > p25_rw),
        np.clip(remaining_work / median_rw, 1.0, 4.0),
        1.0
    )
    energy_scaled = min_incremental_energy * energy_work_weight
    energy_norm = robust_normalize(energy_scaled)
    
    # Starvation penalty: adaptive, bounded, and multi-factor gated
    p95_wait = np.percentile(ready_wait_time, 95, method='midpoint') if N > 1 else np.max(ready_wait_time)
    p95_wait = np.maximum(p95_wait, eps)
    wait_ratio = np.clip(ready_wait_time / p95_wait, 0.0, 1.0)
    # Lateness pressure factor: only active when slack negative; scaled by duration-normalized lateness
    lateness_pressure = np.clip(np.maximum(0.0, -rel_slack), 0.0, 2.0)
    # Work relevance factor: avoid penalizing trivial work
    work_relevance = np.clip(remaining_work / median_rw, 0.1, 5.0)
    starvation_penalty = wait_ratio * lateness_pressure * work_relevance * 0.3
    starvation_penalty = np.clip(starvation_penalty, 0.0, 0.3)
    
    # Time cost (sqrt of exec+comm) and uncertainty normalization
    time_cost = np.sqrt(np.maximum(min_exec_time, eps)) + np.sqrt(np.maximum(min_comm_time, eps))
    time_norm = robust_normalize(time_cost)
    unc_norm = robust_normalize(uncertainty)
    work_norm = robust_normalize(remaining_work)
    
    # Final score: minimize → prioritize
    # Strong negative weights for urgency and criticality-efficiency; positive for others
    score = (
        -4.0 * deadline_urgency 
        - 2.2 * crit_eff_norm 
        + 0.4 * time_norm 
        + 0.3 * energy_norm 
        + 0.25 * work_norm 
        + 0.2 * unc_norm 
        + 0.3 * starvation_penalty
        + 0.15 * norm_energy_eff  # slight positive weight: higher harmonic efficiency = lower priority score
    )
    
    # Final sanitization
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    return score
