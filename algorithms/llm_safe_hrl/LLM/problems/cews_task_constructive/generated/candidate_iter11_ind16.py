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
    v2: Hybrid deadline-energy-fairness priority with robust normalization and adaptive gating.
    
    Key innovations:
    - Deadline urgency: piecewise-linear (Parent 2) + hard penalty for slack <= 0 (Parent 1) → strict DDL enforcement
    - Energy efficiency: critical-energy density (upward_rank * remaining_work / exec_effort) gated by smooth sigmoid on slack (Parent 2), but scaled to dominate only when slack > 0
    - Fairness: wait_ratio = ready_wait_time / (min_exec_time + min_comm_time + eps), always active and normalized via percentile (Parent 2), enhanced with aging boost proportional to |slack|⁻¹ under positive slack
    - Uncertainty: active only in tight-margin regime (0 < slack <= median_positive_slack AND uncertainty > 0.1), then percentile-bounded (Parent 1 + Parent 2 hybrid)
    - All normalizations use robust percentile fallbacks for N=1/flat arrays; no NaN/inf propagation
    - Final score prioritizes: (1) deadline compliance, (2) energy efficiency under margin, (3) fairness, (4) risk mitigation
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
    
    # Deadline urgency: piecewise-linear + hard penalty for slack <= 0
    urgency_raw = np.where(slack <= 0, -slack * 2.0 + 0.5, slack * 0.2)
    norm_urgency = robust_normalize(urgency_raw)
    urgency_term = -4.0 * norm_urgency
    
    # Criticality: upward_rank * remaining_work normalized by execution effort
    exec_effort = np.maximum(min_exec_time, eps)
    critical_density = upward_rank * (remaining_work / exec_effort)
    norm_critical = robust_normalize(critical_density)
    critical_term = -1.5 * norm_critical
    
    # Energy efficiency: marginal energy per criticality, gated by slack-aware sigmoid
    energy_per_crit = min_incremental_energy / (critical_density + eps)
    slack_gate = 0.1 + 0.9 / (1.0 + np.exp(-slack / 1.0))
    energy_gated = energy_per_crit * slack_gate
    norm_energy = robust_normalize(energy_gated)
    efficiency_term = -1.0 * norm_energy
    
    # Fairness: wait_ratio always active, enhanced with slack-aware aging boost
    duration_estimate = min_exec_time + min_comm_time + eps
    wait_ratio = np.clip(ready_wait_time / duration_estimate, 0.0, 10.0)
    norm_wait = robust_normalize(wait_ratio)
    # Aging boost increases with wait time and inversely with positive slack magnitude
    slack_abs_pos = np.where(slack > 0, slack + eps, 1.0)
    aging_boost = np.clip(wait_ratio / slack_abs_pos, 0.0, 5.0)
    norm_aging = robust_normalize(aging_boost)
    fairness_term = -0.5 * np.clip(norm_aging, -1.5, 2.0)
    
    # Uncertainty: active only in tight-margin regime (0 < slack <= median_positive_slack AND uncertainty > 0.1)
    positive_slack_mask = slack > 0
    median_positive_slack = np.median(slack[positive_slack_mask]) if np.any(positive_slack_mask) else np.max(np.abs(slack)) + eps
    uncertainty_active = positive_slack_mask & (slack <= median_positive_slack + eps) & (uncertainty > 0.1)
    uncertainty_gated = np.where(uncertainty_active, uncertainty, 0.0)
    unc_q75 = np.percentile(uncertainty_gated, 75) + eps
    norm_uncertainty = np.clip(uncertainty_gated / unc_q75, 0.0, 3.0)
    uncertainty_term = 0.5 * norm_uncertainty
    
    # Combine terms: urgency dominates, then criticality, efficiency, fairness, uncertainty
    score = urgency_term + critical_term + efficiency_term + fairness_term + uncertainty_term
    
    # Ensure finite deterministic output
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    return score
