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
    Priority rule combining deadline-hardness enforcement, risk-aware energy minimization,
    and starvation-avoiding fairness. Key improvements:
      - Hybrid urgency gating: uses both hard (slack <= 0) and soft (slacks below 30th percentile) modes
      - Criticality-energy ratio normalized via IQR, then dampened by uncertainty only when slack is tight
      - Sigmoid starvation bonus with dynamic threshold based on median wait time (not fixed offset)
      - Robust time-cost metric: sqrt(exec) + sqrt(comm) + eps, scaled robustly
      - Explicit penalty for negative slack (additive, not multiplicative) to enforce hard DDL compliance
      - All operations guarded against NaN/inf/zero; outputs finite (N,) array with deterministic ordering
    """
    eps = 1e-08
    # Ensure float arrays, no in-place modification
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    
    N = len(slack)
    if N == 0:
        return np.array([], dtype=float)
    
    # Robust IQR-based normalization function
    def robust_normalize(x):
        q1, q3 = np.quantile(x, [0.25, 0.75])
        iqr = q3 - q1 + eps
        center = np.median(x)
        return (x - center) / iqr
    
    # --- Deadline Urgency: Hard + Soft gating ---
    # Hard gate: tasks violating or at risk of missing deadline
    is_hard_urgent = (slack <= 0.0).astype(float)
    # Soft gate: top 30% most urgent (by slack), including negative ones
    slack_sorted = np.sort(slack)
    slack_thresh = slack_sorted[max(0, int(0.3 * N))] if N > 0 else 0.0
    is_soft_urgent = (slack <= slack_thresh).astype(float)
    
    # Additive penalty for lateness risk: large linear penalty for slack < 0, capped soft penalty otherwise
    lateness_penalty = np.maximum(0.0, -slack) * 4.0  # Strong hard-DDL enforcement
    soft_urgency = np.where(is_soft_urgent, 
                           1.0 / (1.0 + np.maximum(0.0, slack_thresh - slack) * 0.2 + eps), 
                           0.0)
    deadline_score = lateness_penalty + soft_urgency * 2.0
    
    # --- Criticality-Energy Tradeoff ---
    # Ratio: higher upward_rank per unit energy → prefer high-impact low-energy tasks
    energy_safe = np.maximum(min_incremental_energy, eps)
    crit_eff_ratio = upward_rank / energy_safe
    crit_eff_norm = robust_normalize(crit_eff_ratio)
    # Uncertainty damping: only reduce criticality weight when slack is tight (hard or soft urgent)
    unc_damp = np.where(is_hard_urgent | is_soft_urgent, 
                       1.0 - np.clip(uncertainty, 0.0, 0.9), 
                       1.0)
    crit_eff_score = -crit_eff_norm * unc_damp  # negative: higher ratio → better score
    
    # --- Starvation Avoidance ---
    # Sigmoid bonus calibrated to median wait time (dynamic fairness threshold)
    wait_scale = np.maximum(np.median(ready_wait_time), eps)
    starvation_bonus = 1.0 / (1.0 + np.exp(-(ready_wait_time / (wait_scale + eps) - 1.5)))
    wait_score = -starvation_bonus  # higher wait → lower score (better priority)
    
    # --- Time-Cost & Work Signaling ---
    time_cost = np.sqrt(np.maximum(min_exec_time, eps)) + np.sqrt(np.maximum(min_comm_time, eps)) + eps
    time_norm = robust_normalize(time_cost)
    work_norm = robust_normalize(remaining_work)
    
    # --- Energy & Uncertainty Signals ---
    energy_norm = robust_normalize(min_incremental_energy)
    # Normalize uncertainty only when relevant (tight slack), else suppress
    unc_norm = robust_normalize(uncertainty) * np.where(is_hard_urgent | is_soft_urgent, 1.0, 0.0)
    
    # --- Final weighted score: smaller = better ---
    # Weights emphasize deadline penalty (dominant), then criticality-energy tradeoff, fairness, and efficiency
    score = (
        1.8 * deadline_score +
        1.0 * crit_eff_score +
        0.7 * wait_score +
        0.5 * time_norm +
        0.4 * energy_norm +
        0.3 * work_norm +
        0.2 * unc_norm
    )
    
    # Ensure finite output: replace NaN/inf with safe large values
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    
    return score
