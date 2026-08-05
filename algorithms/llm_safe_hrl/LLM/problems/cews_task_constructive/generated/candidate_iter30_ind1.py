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
    v2 priority rule: Lateness-aware urgency scaling + adaptive slack sensitivity + uncertainty-robust energy prioritization.
      - Urgency now scales *linearly with lateness deficit* (not binary), preserving monotonicity and gradient flow
      - Replaced quadratic uncertainty scaling with calibrated linear+exponential hybrid to avoid over-penalization
      - Slack sensitivity uses smooth, differentiable sigmoid with dynamic scale tied to critical path length
      - Energy inversion activated under *marginal slack* (slack > 0.1 * duration) for fine-grained tradeoff control
      - Work consolidation now incorporates uncertainty-weighted throughput density (work / effective_duration)
      - Relative age normalized by *global* critical path estimate (not per-task) for consistent starvation pressure
      - All components rigorously clipped, nan-cleaned, and MAD-normalized with strict finite-domain guarantees
      - Lateness penalty applied *additively* and unclipped to dominate scheduling when deadline violation looms
    '''
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
    
    def robust_mad_norm(x):
        """MAD-based robust normalization with N=1 safety, finite clipping, and fallback percentile scaling"""
        x_clean = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if N == 1:
            return np.zeros_like(x_clean)
        med = np.median(x_clean)
        abs_dev = np.abs(x_clean - med)
        mad = np.median(abs_dev)
        if mad < eps:
            x_finite = x_clean[np.isfinite(x_clean)]
            if x_finite.size == 0:
                return np.zeros_like(x_clean)
            p05 = np.percentile(x_finite, 5.0, method='midpoint')
            p95 = np.percentile(x_finite, 95.0, method='midpoint')
            x_clipped = np.clip(x_clean, p05, p95)
            x_min = np.min(x_clipped)
            x_max = np.max(x_clipped)
            if x_max - x_min < eps:
                return np.zeros_like(x_clean)
            return (x_clipped - x_min) / (x_max - x_min + eps)
        z = (x_clean - med) / (mad + eps)
        return np.clip(z, -3.0, 3.0)
    
    # Base duration: execution + communication, with epsilon guard
    duration = min_exec_time + min_comm_time + eps
    
    # Lateness severity: linear deficit scaled by duration for interpretable urgency
    slack_deficit = np.maximum(-slack, 0.0)
    norm_deficit = np.divide(slack_deficit, duration, out=np.zeros_like(slack_deficit), where=duration != 0)
    norm_deficit = np.nan_to_num(norm_deficit, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Adaptive slack sensitivity: sigmoid centered at zero, width scales with critical path length
    total_cp = np.max(upward_rank) if N > 0 else 1.0
    cp_scale = np.clip(total_cp, eps, 1e6)
    slack_norm = np.divide(slack, duration + eps, out=np.zeros_like(slack), where=duration != 0)
    slack_norm = np.nan_to_num(slack_norm, nan=0.0)
    # Sigmoid with dynamic width: steeper for short CP, flatter for long CP
    slack_sensitivity = 0.5 * (np.tanh(-slack_norm / (0.15 * cp_scale + eps)) + 1.0)
    
    # Criticality × slack sensitivity — decoupled from energy layer
    critical_slack_sensitivity = upward_rank * slack_sensitivity
    
    # Uncertainty-aware effective duration: linear base + bounded exponential tail to avoid explosion
    # Uses soft cap: exp(min(uncertainty, 3.0)) prevents runaway penalties
    capped_uncertainty = np.clip(uncertainty, 0.0, 3.0)
    exp_unc = np.exp(capped_uncertainty) - 1.0  # [0, ~19.1] → avoids NaN/inf for large uncertainty
    scaled_uncertainty = 0.3 * uncertainty + 0.7 * exp_unc
    effective_duration = duration * (1.0 + np.clip(scaled_uncertainty, 0.0, 5.0) + eps)
    
    # Risk-adjusted energy density: marginal energy per effective time unit
    risk_energy_density = np.divide(min_incremental_energy, effective_duration, 
                                   out=np.zeros_like(min_incremental_energy), where=effective_duration != 0)
    risk_energy_density = np.nan_to_num(risk_energy_density, nan=0.0, posinf=0.0, neginf=0.0)
    norm_energy_density = robust_mad_norm(risk_energy_density)
    
    # Marginal slack gating: invert energy only when slack > 10% of duration (smooth threshold)
    margin_threshold = 0.1 * duration
    is_marginal_safe = (slack > margin_threshold).astype(np.float64)
    energy_score = is_marginal_safe * (-0.38 * norm_energy_density)
    
    # Uncertainty-weighted work density: remaining_work / effective_duration, normalized
    work_density_raw = np.divide(remaining_work, effective_duration + eps, 
                                out=np.zeros_like(remaining_work), where=effective_duration != 0)
    work_density_raw = np.nan_to_num(work_density_raw, nan=0.0, posinf=0.0, neginf=0.0)
    norm_work_density = robust_mad_norm(work_density_raw)
    work_consolidation_bonus = is_marginal_safe * (-0.09 * norm_work_density)
    
    # Starvation pressure: relative age w.r.t. *global* critical path length estimate
    global_cp_length = np.max(duration * upward_rank) if N > 0 else 1.0
    global_cp_length = max(global_cp_length, eps)
    relative_age = np.divide(ready_wait_time, global_cp_length + eps, 
                            out=np.zeros_like(ready_wait_time), where=global_cp_length != 0)
    relative_age = np.nan_to_num(relative_age, nan=0.0, posinf=0.0, neginf=0.0)
    relative_age = np.clip(relative_age, 0.0, 10.0)
    norm_relative_age = robust_mad_norm(relative_age)
    wait_pressure = 0.14 * norm_relative_age  # Apply uniformly (urgency already handled via deficit)
    
    # CP importance: normalized upward rank for structural weighting
    cp_importance = np.divide(upward_rank, cp_scale + eps, 
                             out=np.zeros_like(upward_rank), where=cp_scale != 0)
    norm_cp_importance = robust_mad_norm(cp_importance)
    
    # Assemble final score: smaller = higher priority
    score = np.full(N, 0.0, dtype=np.float64)
    
    # Lateness dominance: unclipped linear penalty dominates all other terms
    score = score + 1000.0 * norm_deficit
    
    # Feasibility-preserving critical path term
    score = score + 0.44 * norm_cp_importance * critical_slack_sensitivity
    
    # Energy and work optimization (only under marginal safety)
    score = score + energy_score
    score = score + work_consolidation_bonus
    
    # Starvation mitigation
    score = score + wait_pressure
    
    # Final safeguard: clip extreme values but preserve ordering; ensure finite & deterministic
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
