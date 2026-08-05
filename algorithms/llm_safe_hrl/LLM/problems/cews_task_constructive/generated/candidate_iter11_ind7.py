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
    Hybrid priority rule combining Parent 2's robust normalization and monotonic urgency
    with Parent 1's criticality-energy coupling and slack-constrained energy scaling.
    Key innovations:
      - Urgency-gated criticality-energy ratio using linear boost (Parent 2) + IQR-stable coupling (Parent 1)
      - Slack-aware energy reversal: low-energy favored when slack > 0; high-criticality/low-energy favored when slack <= 0
      - Robust starvation control via smoothed 90th-percentile wait pressure, gated by urgency and slack sign
      - Uncertainty-weighted energy penalty scaled by slack tightness (Parent 1's risk_exponent logic, simplified)
      - All operations eps-protected, finite-clipped, deterministic, and shape-compliant
      - Uses robust_minmax_norm for stability across all N (Parent 2), but applies IQR normalization only where needed for ordinal preservation
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
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)

    def robust_minmax_norm(x):
        x_min, x_max = np.min(x), np.max(x)
        if x_max - x_min < eps:
            return np.zeros_like(x, dtype=float)
        return (x - x_min) / (x_max - x_min + eps)

    total_latency = min_exec_time + min_comm_time + eps
    median_duration = np.median(total_latency) + eps
    median_slack = np.median(slack)
    
    # Linear slack-gated criticality boost (Parent 2)
    slack_gap = median_slack - slack
    boost_factor = np.clip(slack_gap / (median_duration + eps), 0.0, 1.0)
    amplified_upward_rank = upward_rank * (1.0 + boost_factor)
    
    # Criticality-energy coupling: ratio scaled by work-aware weight (Parent 1 inspired)
    median_ur = np.median(upward_rank) + eps
    ur_ratio = upward_rank / median_ur
    median_rw = np.median(remaining_work) + eps
    rw_ratio = np.clip(remaining_work / median_rw, 0.1, 10.0)
    energy_work_weight = np.where(ur_ratio > 1.5, np.clip(rw_ratio, 1.0, 3.0), 1.0)
    energy_scaled = np.maximum(min_incremental_energy * energy_work_weight, eps)
    
    # Criticality-energy ratio with clipping for stability
    crit_energy_ratio = amplified_upward_rank / energy_scaled
    crit_energy_ratio = np.clip(crit_energy_ratio, eps, 1e6)
    norm_crit_energy_ratio = robust_minmax_norm(crit_energy_ratio)
    
    # Energy reversal: favor low energy when slack > 0, favor high criticality/low energy when slack <= 0
    norm_energy = robust_minmax_norm(min_incremental_energy)
    energy_reversed = np.where(slack > 0, norm_energy, 1.0 - norm_energy)
    
    # Uncertainty-weighted energy penalty, slack-constrained (Parent 1 style, simplified)
    risk_exponent = np.clip(1.0 + 0.3 * np.maximum(0.0, -slack), 1.0, 2.5)
    energy_risk_weighted = min_incremental_energy * np.power(1.0 + uncertainty, risk_exponent)
    energy_penalty = robust_minmax_norm(np.maximum(energy_risk_weighted, eps))
    
    # Latency term
    norm_latency = robust_minmax_norm(total_latency)
    
    # Starvation guard: smoothed 90th-percentile wait pressure, gated by non-urgent & slack-positive tasks
    wait_threshold = np.quantile(ready_wait_time, 0.9, method='midpoint') + eps
    wait_pressure = np.clip(ready_wait_time / (wait_threshold + eps), 0.0, 1.0)
    is_non_urgent_and_safe = (slack > 0).astype(float) * (np.mean(slack > 0) > 0.1).astype(float)
    starvation_term = 1.0 - wait_pressure * is_non_urgent_and_safe
    
    # Work impact scaled by slack sensitivity (Parent 2)
    tau = np.maximum(np.abs(median_slack), 1.0) + eps
    slack_sensitivity = np.exp(-np.clip(np.maximum(-slack, 0.0), 0.0, 100.0) / tau)
    norm_work = robust_minmax_norm(remaining_work)
    work_penalty = norm_work * slack_sensitivity
    
    # Final score: urgency dominant, energy fairness strengthened, latency & starvation balanced
    # Weights tuned to emphasize deadline feasibility first, then energy efficiency, then fairness
    base_urgency = np.where(slack <= 0, 1.0, np.clip(1.0 - slack / (median_duration + eps), 0.0, 1.0))
    score = (
        0.48 * base_urgency +
        0.24 * energy_reversed +
        0.10 * norm_latency +
        0.09 * (1.0 - norm_crit_energy_ratio) +
        0.06 * starvation_term +
        0.03 * work_penalty
    )
    
    # Ensure finite output
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    return score
