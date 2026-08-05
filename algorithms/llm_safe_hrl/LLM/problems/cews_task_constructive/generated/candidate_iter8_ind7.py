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
    Hybrid priority rule: sigmoid urgency + harmonic energy-work efficiency +
    proactive uncertainty amplification + starvation-aware fairness with adaptive wait boost.
    
    Key innovations:
    - Sigmoid urgency (v2) for smooth near-deadline sensitivity, replacing clipped exponential.
    - Harmonic energy-work ratio (v2) for outlier-resilient efficiency scoring.
    - Uncertainty amplification only on late tasks (slack <= 0), but with *normalized* exponent using IQR-scaled lateness.
    - Starvation boost activated when (wait_time > p95) AND (remaining_work > median), scaled by work-normalized urgency gap.
    - Unified robust normalization using median-IQR with fallback and tighter clipping [-7, 7].
    - All divisions eps-protected; nan/inf replaced deterministically; no side effects.
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
        iqr = np.where(iqr < eps, 1.0, iqr)
        norm = (x - med) / iqr
        return np.clip(norm, -7.0, 7.0)
    
    # Task duration baseline for relative urgency scaling
    task_min_duration = np.maximum(min_exec_time + min_comm_time, eps)
    
    # Sigmoid urgency: smooth ramp for near-deadline risk (v2 strength)
    normalized_lateness = -slack / (task_min_duration + eps)
    deadline_urgency = 1.0 / (1.0 + np.exp(-normalized_lateness))
    
    # Proactive uncertainty amplification: only for late tasks, exponent based on normalized lateness magnitude
    lateness_magnitude = np.abs(slack) / (task_min_duration + eps)
    norm_lateness_magnitude = robust_normalize(lateness_magnitude)
    # Exponent grows with normalized lateness but capped to prevent explosion
    risk_exponent = np.clip(1.0 + 0.6 * np.maximum(norm_lateness_magnitude, 0.0), 1.0, 3.5)
    is_late = (slack <= 0.0)
    amp_factor = np.where(is_late, np.power(1.0 + uncertainty, risk_exponent), 1.0)
    
    # Energy-risk-weighted criticality
    energy_risk_weighted = min_incremental_energy * amp_factor
    energy_safe = np.maximum(energy_risk_weighted, eps)
    
    # Harmonic energy-work efficiency: stable under low-energy/low-work outliers (v2 strength)
    energy_per_work = min_incremental_energy / (remaining_work + eps)
    work_efficiency_ratio = remaining_work / (min_incremental_energy + eps)
    harmonic_eff = 2.0 / (1.0 / (energy_per_work + eps) + 1.0 / (work_efficiency_ratio + eps))
    harmonic_eff = np.clip(harmonic_eff, 1e-06, 1e6)
    norm_energy_eff = robust_normalize(harmonic_eff)
    
    # Criticality-efficiency ratio with amplified rank
    amplified_upward_rank = upward_rank * amp_factor
    crit_eff_ratio = amplified_upward_rank / energy_safe
    crit_eff_ratio = np.clip(crit_eff_ratio, 1e-06, 1e6)
    crit_eff_norm = robust_normalize(crit_eff_ratio)
    
    # Starvation guard: activate only for genuinely starved high-work tasks
    p95_wait = np.percentile(ready_wait_time, 95, method='midpoint') if N > 1 else np.max(ready_wait_time)
    median_work = np.median(remaining_work) + eps
    starvation_cond = (ready_wait_time > p95_wait + eps) & (remaining_work > median_work)
    # Boost scales with both wait time and urgency gap (how much urgency drops if delayed further)
    urgency_gap = np.maximum(0.0, deadline_urgency - 0.5)  # Only meaningful for mid-to-high urgency
    starvation_boost = np.where(starvation_cond, 
                               robust_normalize(ready_wait_time) * (1.0 - urgency_gap), 
                               0.0)
    
    # Time cost and resource features
    time_cost = np.sqrt(np.maximum(min_exec_time, eps)) + np.sqrt(np.maximum(min_comm_time, eps))
    time_norm = robust_normalize(time_cost)
    unc_norm = robust_normalize(uncertainty)
    work_norm = robust_normalize(remaining_work)
    
    # Final score: smaller = higher priority
    # Weights tuned to emphasize urgency and criticality-efficiency, while penalizing uncertainty and waiting
    score = (
        -4.2 * deadline_urgency 
        - 2.1 * crit_eff_norm 
        + 0.45 * norm_energy_eff 
        + 0.3 * time_norm 
        + 0.2 * work_norm 
        + 0.22 * unc_norm 
        + 0.3 * starvation_boost
    )
    
    # Deterministic NaN/inf handling
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    return score
