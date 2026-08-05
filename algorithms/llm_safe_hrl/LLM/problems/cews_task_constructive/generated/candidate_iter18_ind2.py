import numpy as np

# 函数名中的 2 由 SeEvo 在生成 Prompt 时替换为目标版本号。
# 八个输入均为长度 N 的一维数组；相同下标始终指向同一个 ready task。
# 返回值也必须是长度 N 的一维数组，并遵守“分数越小，优先级越高”。
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
    Self-evolved v2: Deadline-strict urgency + risk-contextual energy leverage + starvation-robust fairness.
    
    Key advances over v1:
    - Replaces linear-soft urgency with *hard-deadline clamp + soft urgency decay*: 
      strict penalty floor at slack=0 (no grace), exponential decay only for slack>0 → eliminates ambiguity at deadline boundary.
    - Refines feasibility gating: uses *slack >= -eps* (not >=0) to include marginally late tasks in energy-aware scheduling, 
      avoiding premature energy-blind prioritization of borderline-late tasks.
    - Introduces *uncertainty-weighted criticality scaling*: multiplies upward_rank by (1 + uncertainty) only when slack <= 0,
      amplifying critical path pressure under high risk — improves responsiveness to cascading delay.
    - Replaces MAD fallback with *quantile-based robust centering* (median + IQR) and adds explicit constant-array guard,
      ensuring deterministic normalization even when all values are identical.
    - Unifies starvation logic: uses *wait-per-work normalized by feasible subset*, not global max, to prevent domination by outliers.
    - Adds *energy-communication synergy term*: product of normalized energy_score and comm_norm, weighted only for urgent+uncertain tasks,
      capturing joint penalty for high-energy & high-communication tasks under risk.
    - All terms strictly sign-consistent: lower score = higher priority; no inversions or domain flips.
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

    # Compute base metrics
    task_min_duration = np.maximum(min_exec_time + min_comm_time, eps)
    median_dur = np.median(task_min_duration) + eps
    median_ur = np.median(upward_rank) + eps
    median_work = np.median(remaining_work) + eps
    median_wait = np.median(ready_wait_time) + eps
    median_unc = np.median(uncertainty) + eps

    # --- Deadline Urgency: Hard clamp at slack=0, soft decay beyond ---
    # For slack <= 0: fixed penalty 2.0 (strict deadline enforcement)
    # For slack > 0: exp(-slack / median_dur), bounded to [0.1, 1.0]
    urgency_hard = np.full(N, 2.0)
    urgency_soft = np.exp(-np.clip(slack / (median_dur + eps), 0.0, 20.0))
    deadline_urgency = np.where(slack <= 0, urgency_hard, np.clip(urgency_soft, 0.1, 1.0))

    # --- Risk-weighted criticality-energy ratio ---
    base_risk = np.maximum(0.0, -slack) / median_dur
    risk_exponent = np.clip(1.0 + 0.5 * base_risk + 0.3 * uncertainty, 1.0, 3.5)
    energy_risk_weighted = min_incremental_energy * np.power(1.0 + uncertainty, risk_exponent)
    energy_safe = np.maximum(energy_risk_weighted, eps)
    # Amplify upward_rank under urgency: (1 + uncertainty) scaling only when slack <= 0
    ur_amplified = np.where(slack <= 0, upward_rank * (1.0 + uncertainty), upward_rank)
    crit_eff_ratio = ur_amplified / energy_safe
    crit_eff_ratio = np.clip(crit_eff_ratio, 1e-06, 1e6)

    # --- Robust normalization: handles constant arrays via quantile fallback ---
    def robust_iqr_norm(x):
        q1, q3 = np.quantile(x, [0.25, 0.75], method='midpoint')
        iqr = q3 - q1 + eps
        center = np.median(x)
        normed = (x - center) / iqr
        # If iqr ~ 0 (all values near identical), use smallest safe scale
        if np.all(np.abs(iqr) < eps):
            normed = np.zeros_like(x)
        return np.clip(normed, -4.0, 4.0)
    norm_crit_eff = robust_iqr_norm(crit_eff_ratio)

    # --- Feasibility-gated energy scoring: slack >= -eps includes marginally late tasks ---
    feasible_mask = slack >= -eps
    energy_score = np.zeros_like(min_incremental_energy)
    if np.any(feasible_mask):
        feasible_energy = min_incremental_energy[feasible_mask]
        # Use median & IQR on feasible subset
        q1_f, q3_f = np.quantile(feasible_energy, [0.25, 0.75], method='midpoint')
        iqr_f = q3_f - q1_f + eps
        center_f = np.median(feasible_energy)
        energy_norm_base = (min_incremental_energy - center_f) / iqr_f
        energy_score = np.where(feasible_mask, -np.clip(energy_norm_base, -4.0, 4.0), 0.0)
    else:
        energy_score = np.zeros(N)

    # --- Communication pressure: only active for urgent+uncertain tasks ---
    comm_to_work_ratio = min_comm_time / (remaining_work + eps)
    urgent_high_uncert_mask = (slack <= 0.0) & (uncertainty > median_unc)
    comm_pressure = np.where(urgent_high_uncert_mask, comm_to_work_ratio * 4.0, comm_to_work_ratio * 0.1)
    comm_norm = robust_iqr_norm(comm_pressure)

    # --- Starvation penalty: normalized by feasible subset's wait-per-work stats ---
    starvation_mask = feasible_mask & (remaining_work > eps)
    wait_per_work = ready_wait_time / (remaining_work + eps)
    if np.any(starvation_mask):
        feasible_wpw = wait_per_work[starvation_mask]
        wpw_center = np.median(feasible_wpw)
        wpw_iqr = np.quantile(feasible_wpw, 0.75) - np.quantile(feasible_wpw, 0.25) + eps
        wpw_normed = (wait_per_work - wpw_center) / wpw_iqr
        starvation_penalty = np.where(starvation_mask, np.clip(wpw_normed, 0.0, 1.0) * 0.4, 0.0)
    else:
        starvation_penalty = np.zeros(N)

    # --- Synergy term: energy-communication coupling under urgency & uncertainty ---
    synergy_mask = (slack <= 0.0) & (uncertainty > median_unc)
    energy_comm_synergy = np.where(synergy_mask, energy_score * comm_norm * 0.3, 0.0)

    # --- Final component normalizations ---
    norm_dur = robust_iqr_norm(task_min_duration)
    norm_work = robust_iqr_norm(remaining_work)
    norm_unc = robust_iqr_norm(uncertainty)

    # --- Weighted linear combination ---
    # Weights sum to 1.0: 0.45 + 0.25 + 0.10 + 0.08 + 0.05 + 0.04 + 0.02 + 0.01 = 1.0
    score = (
        0.45 * deadline_urgency +
        0.25 * (1.0 - norm_crit_eff) +
        0.10 * norm_dur +
        0.08 * energy_score +
        0.05 * comm_norm +
        0.04 * norm_work +
        0.02 * norm_unc +
        0.01 * starvation_penalty +
        0.0 * synergy_mask  # placeholder for future expansion; currently unused but reserved
    )
    # Add synergy as additive term, not weight-adjusted
    score += synergy_mask * 0.05 * energy_comm_synergy

    # Clean NaN/inf and clamp
    score = np.nan_to_num(score, nan=1e6, posinf=1e6, neginf=-1e6)
    score = np.clip(score, -1e6, 1e6)

    # Ensure shape compliance
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
