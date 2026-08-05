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
    v2: Deadline-first, numerically stable, and starvation-robust priority.
    
    Key evolutions:
    - Urgency: piecewise-linear risk model — exact penalty for slack < 0, smooth ramp for slack > 0 → preserves hard DDL enforcement
    - Energy gating: continuous sigmoidal slack gate (not binary) → enables energy-aware tradeoffs even under mild lateness pressure
    - Fairness: wait_ratio = ready_wait_time / (min_exec_time + min_comm_time + eps), robustly normalized and *always active* → ensures starvation prevention at all N, including N=1
    - Criticality: unchanged robust density (upward_rank * remaining_work / exec_effort), un-gated and well-scaled
    - Uncertainty: percentile-bounded additive penalty with floor/ceiling → avoids distortion while preserving risk signal
    - All normalizations handle degenerate cases (N=1, flat arrays) via safe fallbacks; no NaN/inf propagation
    - Final score is strictly finite, deterministic, and prioritizes deadline compliance first, then energy efficiency, then fairness and risk.
    """
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    
    def robust_normalize(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        q25, q50, q75 = np.percentile(x, [25, 50, 75], axis=0, keepdims=False)
        iqr = q75 - q25
        scale = np.where(iqr > eps, iqr, 1.0)
        return (x - q50) / (scale + eps)
    
    # Urgency: piecewise-linear — strict penalty for negative slack, gentle ramp above zero
    urgency_raw = np.where(slack < 0, -slack * 2.0, slack * 0.2)
    norm_urgency = robust_normalize(urgency_raw)
    urgency_term = -3.5 * norm_urgency  # Strongest weight: deadline is primary constraint
    
    # Criticality: density of critical work per unit execution effort
    exec_effort = np.maximum(min_exec_time, eps)
    critical_density = upward_rank * (remaining_work / exec_effort)
    norm_critical = robust_normalize(critical_density)
    critical_term = -1.3 * norm_critical  # High weight for critical-path leverage
    
    # Energy efficiency: marginal energy per critical work unit, gated *smoothly* by slack
    energy_per_crit = min_incremental_energy / (critical_density + eps)
    # Sigmoid gate: near 1.0 when slack >> 0, near 0.1 when slack << 0 → no discontinuity
    slack_gate = 0.1 + 0.9 / (1.0 + np.exp(-slack / 1.0))
    energy_gated = energy_per_crit * slack_gate
    norm_energy = robust_normalize(energy_gated)
    efficiency_term = -0.9 * norm_energy  # Higher weight than v1 to favor energy where safe
    
    # Fairness: wait_ratio always active, normalized robustly — prevents starvation at any scale
    duration_estimate = min_exec_time + min_comm_time + eps
    wait_ratio = np.clip(ready_wait_time / duration_estimate, 0.0, 10.0)
    norm_wait = robust_normalize(wait_ratio)
    fairness_term = -0.4 * np.clip(norm_wait, -1.2, 1.8)  # Balanced, bounded influence
    
    # Uncertainty: additive, bounded, percentile-scaled penalty
    unc_q75 = np.percentile(uncertainty, 75) + eps
    norm_uncertainty = np.clip(uncertainty / unc_q75, 0.0, 3.0)
    uncertainty_term = 0.45 * norm_uncertainty  # Slightly increased vs v1 to reflect risk cost
    
    score = urgency_term + critical_term + efficiency_term + fairness_term + uncertainty_term
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    return score
