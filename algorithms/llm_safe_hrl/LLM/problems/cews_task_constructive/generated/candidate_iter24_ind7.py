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
    v3 priority rule: Hard deadline lockstep + adaptive urgency gating + 
    risk-aware energy density + starvation-robust fairness + 
    criticality-preserving slack pressure + outlier-resilient normalization.

    Key self-evolution improvements:
    - Replaces fixed percentile gates with *dynamic urgency bands*: 
      urgent (slack <= 0), critical (0 < slack <= median_slack), relaxed (else).
      Enables graded response without arbitrary thresholds.
    - Introduces *risk-adjusted energy density*: 
      min_incremental_energy / (duration * (1 + uncertainty)^2) — stronger penalty for high uncertainty,
      preventing premature selection of high-risk low-energy options near deadlines.
    - Upgrades fairness to *latency-aware starvation guard*: 
      wait_efficiency penalized only in relaxed band, and scaled by (1 - norm_slack) to avoid over-penalizing long-waiting tasks when slack is tight.
    - Replaces heuristic slack_pressure with *critical-path slack sensitivity*: 
      (upward_rank / total_critical_path) * (1 - norm_slack_in_band), preserving structural importance under uncertainty.
    - Adds *workload-aware uncertainty coupling*: uncertainty weighted by normalized remaining_work to prioritize consolidation of heavy sub-DAGs on stable VMs.
    - All features use robust_zscore with MAD; weights sum to 1.0: 
      urgency (0.44) > energy-gated (0.21) > criticality (0.15) > fairness (0.09) > 
      cp_slack_sensitivity (0.07) > unc_work_density (0.04).
    - Strict finite sanitization, deterministic, no in-place mutation, shape-checked.
    """
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=np.float64)
    min_comm_time = np.asarray(min_comm_time, dtype=np.float64)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slack = np.asarray(slack, dtype=np.float64)
    upward_rank = np.asarray(upward_rank, dtype=np.float64)
    remaining_work = np.asarray(remaining_work, dtype=np.float64)
    ready_wait_time = np.asarray(ready_wait_time, dtype=np.float64)
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    inputs = [min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty]
    cleaned = [np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0) for arr in inputs]
    min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty = cleaned
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

    # Band-based urgency classification (no fixed percentiles → adapts to current slack distribution)
    slack_clean = np.nan_to_num(slack, nan=0.0, posinf=0.0, neginf=0.0)
    median_slack = np.median(slack_clean) if N > 0 else 0.0
    is_urgent = (slack_clean <= 0.0).astype(np.float64)
    is_critical = ((slack_clean > 0.0) & (slack_clean <= median_slack)).astype(np.float64)
    is_relaxed = (slack_clean > median_slack).astype(np.float64)

    urgency_score = np.full(N, -1000000000000000.0, dtype=np.float64)
    
    duration = min_exec_time + min_comm_time + eps
    slack_deficit = np.maximum(-slack_clean, 0.0)
    norm_urgency = np.divide(slack_deficit, duration + eps, out=np.zeros_like(slack_deficit), where=duration + eps != 0)
    norm_urgency = np.clip(norm_urgency, 0.0, 1.0)

    # Risk-adjusted energy density: quadratic uncertainty penalty for stability-critical decisions
    effective_duration = duration * (1.0 + uncertainty + eps) ** 2
    risk_adj_energy_density = np.divide(min_incremental_energy, effective_duration, 
                                        out=np.zeros_like(min_incremental_energy), 
                                        where=effective_duration != 0)
    risk_adj_energy_density = np.where(np.isfinite(risk_adj_energy_density), risk_adj_energy_density, 0.0)
    norm_energy = robust_zscore(risk_adj_energy_density)

    # Criticality: upward_rank normalized w.r.t. total critical path estimate (proxy: max upward_rank)
    total_cp = np.max(upward_rank) if N > 0 else 1.0
    cp_importance = np.divide(upward_rank, total_cp + eps, out=np.zeros_like(upward_rank), where=total_cp + eps != 0)
    norm_cp_importance = robust_zscore(cp_importance)

    # Fairness: only active in relaxed band, and attenuated by slack proximity to avoid starving tasks near deadline
    wait_efficiency = np.divide(ready_wait_time, remaining_work + eps, 
                               out=np.zeros_like(ready_wait_time), 
                               where=remaining_work + eps != 0)
    wait_efficiency = np.where(np.isfinite(wait_efficiency), wait_efficiency, 0.0)
    norm_wait_efficiency = robust_zscore(wait_efficiency)
    # Scale fairness penalty by how far slack is from critical band → less penalty when slack tight
    norm_slack_in_band = np.divide(np.clip(slack_clean, 0.0, None), median_slack + eps, 
                                   out=np.ones_like(slack_clean), where=median_slack + eps != 0)
    fairness_mask = is_relaxed * (1.0 - np.clip(norm_slack_in_band, 0.0, 1.0))

    # Critical-path slack sensitivity: combines structural importance and slack pressure
    # Higher weight for high-upward-rank tasks *only* when slack is tight (i.e., in urgent/critical bands)
    slack_pressure_weight = is_urgent + is_critical * 0.5  # full weight for urgent, half for critical
    cp_slack_sensitivity = cp_importance * slack_pressure_weight
    norm_cp_slack_sensitivity = robust_zscore(cp_slack_sensitivity)

    # Uncertainty-work density: prioritize stabilizing high-work sub-DAGs on low-uncertainty VMs
    unc_work_density = uncertainty * (remaining_work / (np.median(remaining_work) + eps) if N > 0 else 1.0)
    unc_work_density = np.where(np.isfinite(unc_work_density), unc_work_density, 0.0)
    norm_unc_work_density = robust_zscore(unc_work_density)

    # Energy gating: only apply energy optimization in critical & relaxed bands (never in urgent — hard DDL first)
    energy_gate = is_critical + is_relaxed

    # Final weighted score — urgency dominates; others refine within feasibility
    w_urgency = 0.44
    w_energy = 0.21
    w_critical = 0.15
    w_fairness = 0.09
    w_cp_slack = 0.07
    w_unc_work = 0.04

    base_score = (
        w_urgency * norm_urgency +
        w_energy * (norm_energy * energy_gate) +
        w_critical * (-norm_cp_importance) +
        w_fairness * (norm_wait_efficiency * fairness_mask) +
        w_cp_slack * (-norm_cp_slack_sensitivity) +
        w_unc_work * norm_unc_work_density
    )

    score = np.where(is_urgent, urgency_score, base_score)
    score = np.clip(score, -1000000000000000.0, 1000000000000000.0)
    score = np.nan_to_num(score, nan=1000000000000000.0, posinf=1000000000000000.0, neginf=-1000000000000000.0)

    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
