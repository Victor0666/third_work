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
    v5 priority rule: Hard-deadline lockstep + unclipped linear urgency + 
    quadratic risk-energy density + starvation-robust monotonic fairness + 
    critical-path-weighted slack deficit + uncertainty-normalized work consolidation +
    MAD-based robust normalization + explicit lateness dominance.

    Key self-evolution improvements:
    - Restores *unclipped linear lateness penalty* as primary urgency signal (not inverse-slack),
      ensuring strict monotonic penalty growth with deficit — fixes weakened DDL pressure.
    - Replaces confidence-modulated inv_slack with *direct slack_deficit scaling* in urgency term,
      weighted by duration-normalized criticality to avoid biasing short tasks.
    - Simplifies fairness to *monotonic wait-efficiency relief*: penalize only in relaxed band,
      but use raw ready_wait_time (not ratio) scaled by normalized remaining_work — improves starvation detection.
    - Strengthens cp_slack_sensitivity: uses upward_rank * slack_deficit * (1 + uncertainty) to prioritize
      high-criticality, high-risk, late tasks — preserves structural importance under uncertainty.
    - Introduces *uncertainty-normalized work consolidation*: uncertainty / (1 + norm_remaining_work + eps)
      to prefer stable VMs for light sub-DAGs and tolerate risk for heavy ones — balances load & reliability.
    - All components use unified robust_zscore (MAD, [-5,5]); weights sum to 1.0:
      urgency (0.45) > energy-gated (0.22) > criticality (0.16) > fairness (0.08) > cp_slack (0.06) > unc_work (0.03).
    - Strict finite sanitization: all divisions guarded, NaN/inf replaced deterministically, no in-place mutation.
    """
    eps = 1e-08
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    slack = np.nan_to_num(np.asarray(slack, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=np.float64)
    
    def robust_zscore(x):
        if x.size == 0:
            return np.zeros_like(x)
        med = np.median(x)
        dev = x - med
        mad = np.median(np.abs(dev))
        if mad < eps:
            return np.zeros_like(x)
        z = dev / (mad + eps)
        return np.clip(z, -5.0, 5.0)
    
    # Dynamic urgency bands: urgent (slack <= 0), critical (0 < slack <= median), relaxed (else)
    median_slack = np.median(slack) if N > 0 else 0.0
    is_urgent = (slack <= 0.0).astype(np.float64)
    is_critical = ((slack > 0.0) & (slack <= median_slack)).astype(np.float64)
    is_relaxed = (slack > median_slack).astype(np.float64)
    
    # Linear lateness penalty dominates urgency: strong monotonic push for overdue tasks
    slack_deficit = np.maximum(-slack, 0.0)
    duration = min_exec_time + min_comm_time + eps
    # Normalize deficit by duration to avoid biasing short tasks; keep linear scale
    norm_deficit = np.divide(slack_deficit, duration, out=np.zeros_like(slack_deficit), where=duration != 0)
    norm_deficit = np.where(np.isfinite(norm_deficit), norm_deficit, 0.0)
    # Urgency score: large negative for urgent tasks → highest priority
    urgency_score = -1000000000000000.0 + 10000.0 * norm_deficit
    
    # Risk-adjusted energy density: quadratic uncertainty penalty on effective duration
    effective_duration = duration * (1.0 + uncertainty + eps) ** 2
    risk_adj_energy_density = np.divide(min_incremental_energy, effective_duration, 
                                        out=np.zeros_like(min_incremental_energy), 
                                        where=effective_duration != 0)
    risk_adj_energy_density = np.where(np.isfinite(risk_adj_energy_density), risk_adj_energy_density, 0.0)
    norm_energy = robust_zscore(risk_adj_energy_density)
    
    # Criticality: upward_rank normalized by total critical path length
    total_cp = np.max(upward_rank) if N > 0 else 1.0
    cp_importance = np.divide(upward_rank, total_cp + eps, out=np.zeros_like(upward_rank), where=total_cp + eps != 0)
    norm_cp_importance = robust_zscore(cp_importance)
    
    # Fairness: monotonic starvation guard — raw wait time relief, gated by relaxed band only
    # Scale by normalized remaining_work to prevent over-penalizing heavy tasks waiting long
    norm_rw = robust_zscore(remaining_work)
    wait_relief = ready_wait_time / (np.clip(norm_rw, 0.1, 10.0) + eps)  # avoid division by near-zero
    wait_relief = np.where(np.isfinite(wait_relief), wait_relief, 0.0)
    norm_wait_relief = robust_zscore(wait_relief)
    fairness_mask = is_relaxed
    
    # Critical-path-aware slack sensitivity: upward_rank * slack_deficit * (1 + uncertainty)
    # Prioritizes high-criticality, overdue, high-risk tasks — preserves structure under pressure
    cp_slack_sensitivity = cp_importance * slack_deficit * (1.0 + uncertainty)
    cp_slack_sensitivity = np.where(np.isfinite(cp_slack_sensitivity), cp_slack_sensitivity, 0.0)
    norm_cp_slack_sensitivity = robust_zscore(cp_slack_sensitivity)
    
    # Uncertainty-normalized work consolidation: favor stable VMs for light work, tolerate risk for heavy
    # Use (1 + norm_remaining_work) in denominator to invert preference: low rw → high weight for stability
    norm_rw_for_unc = np.clip(norm_rw, 0.0, 10.0)  # prevent extreme inversion
    unc_work_density = uncertainty / (1.0 + norm_rw_for_unc + eps)
    unc_work_density = np.where(np.isfinite(unc_work_density), unc_work_density, 0.0)
    norm_unc_work_density = robust_zscore(unc_work_density)
    
    # Energy gating: only apply energy density scoring in non-urgent bands
    energy_gate = is_critical + is_relaxed
    
    # Weight allocation sums to 1.0; urgency increased to enforce hard deadline adherence
    w_urgency = 0.45
    w_energy = 0.22
    w_critical = 0.16
    w_fairness = 0.08
    w_cp_slack = 0.06
    w_unc_work = 0.03
    
    # Base score combines all components; fairness and cp_slack are negative (higher priority = lower score)
    base_score = (
        w_urgency * -norm_deficit +  # now directly uses normalized deficit (linear, monotonic)
        w_energy * (norm_energy * energy_gate) +
        w_critical * -norm_cp_importance +
        w_fairness * (-norm_wait_relief * fairness_mask) +  # negative because wait_relief should reduce priority score
        w_cp_slack * -norm_cp_slack_sensitivity +
        w_unc_work * norm_unc_work_density
    )
    
    # Apply urgent penalty: overrides base score for overdue tasks
    score = np.where(is_urgent, urgency_score, base_score)
    
    # Final sanitization: clip and replace NaN/inf
    score = np.clip(score, -1000000000000000.0, 1000000000000000.0)
    score = np.nan_to_num(score, nan=1000000000000000.0, posinf=1000000000000000.0, neginf=-1000000000000000.0)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
