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
    Hybrid priority rule combining Parent 1's hard urgency gating and fairness guarantees
    with Parent 2's relative slack modeling, dynamic risk exponent, and IQR robustness.
    
    Key innovations:
    - Hard urgency gate (binary) for strict DDL compliance, *combined* with smooth relative slack for non-urgent tasks
    - Unified criticality-energy term: upward_rank / (min_incremental_energy * (1+uncertainty)^alpha), normalized via IQR
    - Adaptive starvation guard: log-scaled wait time weighted by slack sign and remaining work, bounded [0, 0.25]
    - Work-aware energy scaling only applied when upward_rank > median (preserves low-criticality energy efficiency)
    - All ratios use eps-protected division and finite clipping; no min-max normalization (IQR only)
    - Final score prioritizes deadline adherence (weight 0.45), risk-aware energy (0.3), latency (0.1), fairness (0.08), uncertainty (0.04), criticality (0.03)
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
    
    # Robust IQR-based symmetric normalization (sign-preserving, outlier-robust)
    def robust_iqr_normalize(x):
        q1, q3 = np.quantile(x, [0.25, 0.75])
        iqr = q3 - q1 + eps
        center = np.median(x)
        normed = (x - center) / iqr
        return np.clip(normed, -10.0, 10.0)
    
    # Task intrinsic duration for relative slack scaling
    task_min_duration = np.maximum(min_exec_time + min_comm_time, eps)
    rel_slack = slack / task_min_duration
    
    # Hard urgency gate: binary flag for slack <= 0 (strict DDL enforcement)
    is_urgent = (slack <= 0.0).astype(float)
    
    # Deadline urgency: hard gate for urgent tasks, smooth relative slack penalty for others
    deadline_urgency = np.where(
        is_urgent == 1.0,
        1.0 + np.maximum(0.0, -rel_slack),  # amplify lateness severity
        1.0 / (1.0 + np.maximum(0.0, rel_slack) * 0.1 + eps)  # soft decay for slack > 0
    )
    
    # Dynamic risk exponent: increase energy penalty under lateness risk
    risk_exponent = np.clip(1.0 + 0.5 * np.maximum(0.0, -slack), 1.0, 3.0)
    energy_risk_weighted = min_incremental_energy * np.power(1.0 + uncertainty, risk_exponent)
    energy_safe = np.maximum(energy_risk_weighted, eps)
    
    # Criticality-energy ratio: higher rank / higher risk-adjusted energy => higher priority
    crit_eff_ratio = upward_rank / energy_safe
    crit_eff_ratio = np.clip(crit_eff_ratio, 1e-06, 1e6)
    crit_eff_norm = robust_iqr_normalize(crit_eff_ratio)
    
    # Work-aware energy scaling: only scale up energy cost for high-criticality tasks
    median_ur = np.median(upward_rank) + eps
    ur_ratio = upward_rank / median_ur
    energy_work_weight = np.where(ur_ratio > 1.5, np.clip(remaining_work / (np.median(remaining_work) + eps), 1.0, 3.0), 1.0)
    energy_scaled = min_incremental_energy * energy_work_weight
    energy_norm = robust_iqr_normalize(energy_scaled)
    
    # Latency term: sqrt-exec + sqrt-comm, normalized
    time_cost = np.sqrt(np.maximum(min_exec_time, eps)) + np.sqrt(np.maximum(min_comm_time, eps))
    time_norm = robust_iqr_normalize(time_cost)
    
    # Uncertainty normalization
    unc_norm = robust_iqr_normalize(uncertainty)
    
    # Starvation guard: log-scaled wait time, adaptively bounded
    wait_max = np.maximum(np.max(ready_wait_time), eps)
    wait_normalized = np.log1p(ready_wait_time) / np.log1p(wait_max)
    # Scale down wait penalty when slack is negative (urgency dominates) or work is trivial
    wait_scale_factor = np.where(
        slack < 0.0,
        0.2,
        np.where(remaining_work < np.median(remaining_work) + eps, 0.1, 0.25)
    )
    wait_guard = wait_normalized * wait_scale_factor
    
    # Final weighted score: smaller = better
    score = (
        0.45 * (1.0 - deadline_urgency) +           # prioritize urgent tasks (lower score when urgent)
        0.30 * energy_norm +                        # lower energy cost → lower score
        0.10 * time_norm +                          # lower latency → lower score
        0.08 * wait_guard +                         # penalize starvation, but bounded
        0.04 * unc_norm +                           # lower uncertainty → lower score
        0.03 * (1.0 - crit_eff_norm)                # higher criticality/energy ratio → lower score
    )
    
    # Clamp and sanitize
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    
    return score
