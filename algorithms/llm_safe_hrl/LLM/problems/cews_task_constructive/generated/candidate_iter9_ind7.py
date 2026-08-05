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

    '''
    Self-evolved priority rule v2: deadline-aware harmonic efficiency + adaptive risk amplification +
    starvation-resolved urgency gap + work-normalized fairness boost.

    Key improvements over v1:
    - Replaces fixed sigmoid urgency with *adaptive steepness*: eta = clip(4.0 + 2.0 * (1.0 - exp(-|slack|/task_min_duration)), 2.0, 6.0)
      → sharper near-deadline response for tight DDLs, gentler for loose ones
    - Harmonic energy-work ratio now uses *fuzzy-work-normalized* denominator: 
      harmonic_eff = 2 / (1/(energy_per_work + eps) + 1/(work_efficiency_ratio + eps)) * (1 + 0.3 * uncertainty)
      → explicitly penalizes high-uncertainty low-efficiency tasks
    - Uncertainty amplification refined: only activates when (slack <= 0) AND (uncertainty > p75), 
      with exponent scaled by normalized lateness *and* uncertainty rank → avoids over-amplification on low-risk late tasks
    - Starvation boost upgraded to *work-relative wait penalty*: 
      boost = (ready_wait_time / (remaining_work + eps)) * robust_normalize(ready_wait_time) 
      activated only when (wait_ratio > p90) → prioritizes high-work-starved tasks more fairly
    - Introduces *critical-path fidelity term*: |upward_rank - median(upward_rank)| / (IQR + eps) → 
      penalizes deviation from critical-path centrality to avoid bias toward extreme ranks
    - Tighter robust normalization with dynamic clipping bounds: [-6.5, 6.5] for most terms, [-5.0, 5.0] for fairness terms
    - All nan/inf replaced via np.nan_to_num with finite bounds; no divisions without eps protection.
    '''
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

    def robust_normalize(x, clip_low=-6.5, clip_high=6.5):
        if x.size == 0:
            return np.zeros_like(x)
        med = np.median(x)
        q1, q3 = np.quantile(x, [0.25, 0.75], method='midpoint')
        iqr = q3 - q1 + eps
        iqr = np.where(iqr < eps, 1.0, iqr)
        norm = (x - med) / iqr
        return np.clip(norm, clip_low, clip_high)

    task_min_duration = np.maximum(min_exec_time + min_comm_time, eps)
    
    # Adaptive sigmoid steepness: steeper for tighter deadlines
    norm_lateness_abs = np.abs(slack) / (task_min_duration + eps)
    eta = np.clip(4.0 + 2.0 * (1.0 - np.exp(-norm_lateness_abs)), 2.0, 6.0)
    deadline_urgency = 1.0 / (1.0 + np.exp(-eta * (-slack) / (task_min_duration + eps)))
    
    # Lateness magnitude for risk scaling
    lateness_magnitude = np.abs(slack) / (task_min_duration + eps)
    norm_lateness_magnitude = robust_normalize(lateness_magnitude, -5.0, 5.0)
    
    # Uncertainty amplification: only for late AND high-uncertainty tasks
    p75_unc = np.percentile(uncertainty, 75, method='midpoint') if N > 1 else np.max(uncertainty)
    is_late_and_risky = (slack <= 0.0) & (uncertainty > p75_unc + eps)
    risk_exponent = np.clip(1.0 + 0.7 * np.maximum(norm_lateness_magnitude, 0.0), 1.0, 3.8)
    amp_factor = np.where(is_late_and_risky, np.power(1.0 + uncertainty, risk_exponent), 1.0)
    
    energy_risk_weighted = min_incremental_energy * amp_factor
    energy_safe = np.maximum(energy_risk_weighted, eps)
    
    # Fuzzy-work-normalized harmonic efficiency
    energy_per_work = min_incremental_energy / (remaining_work + eps)
    work_efficiency_ratio = remaining_work / (min_incremental_energy + eps)
    harmonic_eff = 2.0 / (1.0 / (energy_per_work + eps) + 1.0 / (work_efficiency_ratio + eps))
    harmonic_eff = harmonic_eff * (1.0 + 0.3 * uncertainty)  # penalize high-uncertainty inefficiency
    harmonic_eff = np.clip(harmonic_eff, 1e-06, 1000000.0)
    norm_energy_eff = robust_normalize(harmonic_eff)
    
    amplified_upward_rank = upward_rank * amp_factor
    crit_eff_ratio = amplified_upward_rank / energy_safe
    crit_eff_ratio = np.clip(crit_eff_ratio, 1e-06, 1000000.0)
    crit_eff_norm = robust_normalize(crit_eff_ratio)
    
    # Work-relative starvation boost: wait_time per unit work, normalized
    wait_per_work = ready_wait_time / (remaining_work + eps)
    p90_wait_ratio = np.percentile(wait_per_work, 90, method='midpoint') if N > 1 else np.max(wait_per_work)
    starvation_cond = wait_per_work > p90_wait_ratio + eps
    starvation_boost = np.where(starvation_cond, 
                               robust_normalize(ready_wait_time, -5.0, 5.0) * 
                               robust_normalize(wait_per_work, -5.0, 5.0), 
                               0.0)
    
    # Critical-path fidelity term: penalize deviation from median upward_rank
    cp_fidelity = np.abs(upward_rank - np.median(upward_rank)) / (np.quantile(upward_rank, 0.75, method='midpoint') - 
                                                                  np.quantile(upward_rank, 0.25, method='midpoint') + eps)
    cp_fidelity_norm = robust_normalize(cp_fidelity, -5.0, 5.0)
    
    time_cost = np.sqrt(np.maximum(min_exec_time, eps)) + np.sqrt(np.maximum(min_comm_time, eps))
    time_norm = robust_normalize(time_cost)
    unc_norm = robust_normalize(uncertainty, -5.0, 5.0)
    work_norm = robust_normalize(remaining_work, -5.0, 5.0)
    
    # Final score: smaller = higher priority
    score = (-4.5 * deadline_urgency 
             - 2.3 * crit_eff_norm 
             + 0.5 * norm_energy_eff 
             + 0.32 * time_norm 
             + 0.18 * work_norm 
             + 0.24 * unc_norm 
             + 0.35 * starvation_boost 
             + 0.15 * cp_fidelity_norm)
    
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    return score
