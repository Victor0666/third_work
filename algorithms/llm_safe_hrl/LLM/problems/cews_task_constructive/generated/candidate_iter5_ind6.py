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
    Hybrid priority rule: integrates Parent 2's relative slack & dynamic risk exponent
    with Parent 1's starvation guard and energy-per-work efficiency, plus novel
    deadline-feasibility gating and uncertainty-aware criticality damping.
    
    Key innovations:
    - Deadline feasibility gate: only penalize urgency *if* task is on critical path
      (upward_rank > median) AND slack <= 0 → prevents over-prioritizing non-critical late tasks
    - Unified energy efficiency: uses both energy_per_work AND work_efficiency_ratio,
      combined via geometric mean for robustness to outliers
    - Uncertainty-damped criticality: downward scales upward_rank by (1 + uncertainty)^beta
      where beta = clip(0.5 * max(0, -slack), 0, 1.5) → reduces importance of uncertain critical tasks
    - Starvation guard upgraded: activates when (slack <= 0) OR (ready_wait_time > p95 AND remaining_work > p25)
      → ensures fairness without rewarding trivial long waits
    - All normalization uses symmetric IQR with explicit zero-variance fallback and tight clipping
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
        norm = (x - med) / iqr
        return np.clip(norm, -10.0, 10.0)
    
    # Relative slack: urgency scaled by intrinsic task duration
    task_min_duration = np.maximum(min_exec_time + min_comm_time, eps)
    rel_slack = slack / task_min_duration
    
    # Deadline feasibility gate: only apply strong urgency penalty if task is critical AND late
    median_ur = np.median(upward_rank) + eps
    is_critical_late = (upward_rank > median_ur) & (slack <= 0.0)
    
    # Urgency: linear penalty for critical late tasks, soft decay otherwise
    urgency_linear = np.where(is_critical_late, np.maximum(0.0, -rel_slack), 0.0)
    urgency_soft = 1.0 / (1.0 + np.maximum(0.0, rel_slack) * 0.15 + eps)
    deadline_urgency = np.where(is_critical_late, 1.0 + urgency_linear, urgency_soft)
    
    # Dynamic risk exponent for energy weighting: amplifies under lateness
    risk_exponent = np.clip(1.0 + 0.5 * np.maximum(0.0, -slack), 1.0, 3.0)
    energy_risk_weighted = min_incremental_energy * np.power(1.0 + uncertainty, risk_exponent)
    energy_safe = np.maximum(energy_risk_weighted, eps)
    
    # Energy efficiency: geometric mean of energy-per-work and work-efficiency-ratio
    # avoids dominance by extreme values in either direction
    energy_per_work = min_incremental_energy / (remaining_work + eps)
    work_efficiency_ratio = remaining_work / (min_incremental_energy + eps)
    energy_efficiency = np.sqrt(np.clip(energy_per_work * work_efficiency_ratio, 1e-06, 1e6))
    norm_energy_eff = robust_normalize(np.clip(energy_efficiency, 1e-06, 1e6))
    
    # Uncertainty-damped criticality: reduce rank weight when uncertainty is high and slack negative
    unc_damp_factor = np.power(1.0 + uncertainty, np.clip(0.5 * np.maximum(0.0, -slack), 0.0, 1.5))
    damped_upward_rank = upward_rank / (unc_damp_factor + eps)
    crit_eff_ratio = damped_upward_rank / energy_safe
    crit_eff_ratio = np.clip(crit_eff_ratio, 1e-06, 1e6)
    crit_eff_norm = robust_normalize(crit_eff_ratio)
    
    # Starvation guard: activate only for non-trivial work + long wait OR any critical lateness
    p95_wait = np.percentile(ready_wait_time, 95, method='midpoint') if N > 1 else 0.0
    p25_work = np.percentile(remaining_work, 25, method='midpoint') if N > 1 else 0.0
    starvation_cond = (slack <= 0.0) | ((ready_wait_time > p95_wait + eps) & (remaining_work > p25_work + eps))
    starvation_penalty = np.where(starvation_cond, robust_normalize(ready_wait_time), 0.0)
    
    # Work-aware time cost: sqrt-normalized execution + comm, normalized
    time_cost = np.sqrt(np.maximum(min_exec_time, eps)) + np.sqrt(np.maximum(min_comm_time, eps))
    time_norm = robust_normalize(time_cost)
    
    # Uncertainty and work normalization
    unc_norm = robust_normalize(uncertainty)
    work_norm = robust_normalize(remaining_work)
    
    # Final score: minimize → prioritize high urgency, high criticality-efficiency, low energy-efficiency,
    # low time cost, low work, low uncertainty; penalize starvation only when justified
    score = (
        -3.5 * deadline_urgency          # strongest pull for feasible deadlines
        - 1.8 * crit_eff_norm             # reward critical + energy-efficient tasks
        + 0.4 * norm_energy_eff          # prefer energy-efficient per-work tasks
        + 0.3 * time_norm                # mild preference for faster tasks
        + 0.2 * work_norm                # slight bias toward smaller workloads
        + 0.15 * unc_norm                # light penalty for high uncertainty
        + 0.2 * starvation_penalty       # controlled starvation mitigation
    )
    
    # Final guard against NaN/inf
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    return score
