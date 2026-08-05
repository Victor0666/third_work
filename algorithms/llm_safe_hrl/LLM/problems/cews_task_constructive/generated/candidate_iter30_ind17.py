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

    eps = 1e-08
    def sanitize(x):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(x, nan=eps, posinf=1e6, neginf=-1e6)
    
    # Sanitize all inputs without in-place mutation
    min_exec_time = sanitize(min_exec_time)
    min_comm_time = sanitize(min_comm_time)
    min_incremental_energy = sanitize(min_incremental_energy)
    slack = sanitize(slack)
    upward_rank = sanitize(upward_rank)
    remaining_work = sanitize(remaining_work)
    ready_wait_time = sanitize(ready_wait_time)
    uncertainty = sanitize(uncertainty)
    
    # Robust slack: subtract uncertainty only when slack > 0 (risk-aware buffer)
    robust_slack = np.where(slack > 0, slack - uncertainty * 0.3, slack)
    
    # Monotonic, bounded urgency: linear ramp from slack=0 (urgency=0) to slack=-tau → urgency=1.0
    # Uses piecewise-linear with explicit clipping; strictly decreasing in slack → deterministic ordering
    tau_urgency = 2.0
    urgency_signal = np.clip((tau_urgency - robust_slack) / (tau_urgency + eps), 0.0, 1.0)
    
    # Hard deadline violation dominance: tasks with robust_slack < -eps get max urgency (1.0)
    # but retain *relative* ordering among violating tasks via scaled slack distance
    violation_mask = robust_slack < -eps
    urgency_final = np.where(violation_mask, 
                           1.0 + (robust_slack / (tau_urgency + eps)),  # monotonic penalty: more negative → higher urgency
                           urgency_signal)
    
    # Latency pressure: weighted by critical path importance and execution+comm cost
    exec_comm_sum = min_exec_time + min_comm_time + eps
    latency_pressure = upward_rank * exec_comm_sum * (1.0 + urgency_final) * np.clip(1.0 - uncertainty * 0.5, 0.3, 1.0)
    
    # Energy efficiency term: penalize high energy-per-work *only when safety margin is low*
    energy_per_work = min_incremental_energy / (remaining_work + eps)
    safety_margin = np.clip(robust_slack / (4.0 + eps), 0.0, 1.0)  # wider safe zone
    energy_term = energy_per_work * (1.0 - safety_margin) * (1.0 + urgency_final * 0.8)
    
    # Critical-path density: work-normalized upstream importance; gated by urgency & uncertainty
    cp_density = (upward_rank + eps) * (remaining_work + eps) / (exec_comm_sum + eps)
    cp_gated = cp_density * (1.0 + urgency_final * 0.6) * np.clip(1.0 - uncertainty * 0.4, 0.4, 1.0)
    
    # Fairness: starvation relief only when safe *and* uncertain — avoids premature boosting
    fairness_base = ready_wait_time * (1.0 + upward_rank) / (exec_comm_sum + eps)
    fairness_boost = np.where((robust_slack > 4.0) & (uncertainty > 0.25), 
                            uncertainty * 0.5 * fairness_base, 0.0)
    fairness_term = fairness_base + fairness_boost
    
    # Adaptive per-term normalization: uses IQR-based scaling for robustness to outliers
    def robust_normalize(x):
        x = np.clip(x, -1e5, 1e5)
        if x.size == 1:
            return np.array([0.0])
        q25, q75 = np.percentile(x, [25, 75])
        iqr = q75 - q25
        if iqr < eps:
            return np.zeros_like(x)
        # Center on median, scale by IQR → preserves relative order better than min-max
        median = np.median(x)
        norm_x = (x - median) / (iqr + eps)
        # Then map to [0,1] via sigmoid-like bounded transform
        return 1.0 / (1.0 + np.exp(-norm_x * 0.5))
    
    norm_urgency = robust_normalize(urgency_final)
    norm_latency = robust_normalize(latency_pressure)
    norm_energy = robust_normalize(energy_term)
    norm_cp = robust_normalize(cp_gated)
    norm_fair = robust_normalize(fairness_term)
    
    # Weighted combination: stronger deadline/latency emphasis, reduced CP weight, fairness as mild correction
    score = (-35.0 * norm_urgency 
             - 20.0 * norm_latency 
             - 12.0 * norm_energy 
             + 1.0 * norm_cp 
             - 0.5 * norm_fair)
    
    # Absolute violation priority: ensure all violating tasks rank *above* all non-violating ones
    # Use deterministic offset: base_min - 1e9 ensures strict ordering, not just large gap
    if np.any(violation_mask):
        non_violating_scores = score[~violation_mask] if np.any(~violation_mask) else score
        base_min = np.min(non_violating_scores)
        score = np.where(violation_mask, base_min - 1e9, score)
    
    # Final sanitization: ensure finite, shape-(N,), deterministic output
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)
    
    # Enforce shape (N,) — never scalar or column vector
    return score.reshape(-1)
