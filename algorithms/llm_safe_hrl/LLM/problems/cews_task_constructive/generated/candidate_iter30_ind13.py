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
        x = np.clip(x, -1000000.0, 1000000.0)
        x = np.nan_to_num(x, nan=eps, posinf=1000000.0, neginf=-1000000.0)
        return x
    min_exec_time = sanitize(min_exec_time)
    min_comm_time = sanitize(min_comm_time)
    min_incremental_energy = sanitize(min_incremental_energy)
    slack = sanitize(slack)
    upward_rank = sanitize(upward_rank)
    remaining_work = sanitize(remaining_work)
    ready_wait_time = sanitize(ready_wait_time)
    uncertainty = sanitize(uncertainty)
    
    # --- Key improvement 1: Dynamic urgency scaling with per-task slack sensitivity ---
    # Use median-based robust scaling instead of mean to reduce outlier influence on τ
    abs_slack_med = np.median(np.abs(slack)) if slack.size > 1 else np.abs(slack)[0]
    tau_urgency = 2.0 + np.clip(abs_slack_med * 0.15, 0.5, 4.0)
    urgency_raw = np.maximum(-slack, 0.0)  # raw lateness penalty
    
    # --- Key improvement 2: Critical-path pressure refined with uncertainty-aware criticality decay ---
    exec_comm_sum = min_exec_time + min_comm_time + eps
    # Weight upward_rank by remaining_work but dampen aggressively under high uncertainty
    cp_density = (upward_rank + eps) * (remaining_work + eps) / (exec_comm_sum + 1.0)
    uncertainty_damp = np.clip(1.0 - uncertainty**1.5, 0.02, 1.0)  # stronger damping for high uncertainty
    latency_pressure = cp_density * uncertainty_damp
    
    # --- Key improvement 3: Safety-aware energy term with hard deadline gating ---
    # Only prioritize energy *when slack is non-negative* — no energy optimization under violation risk
    safety_mask = (slack >= 0.0).astype(float)
    energy_term = min_incremental_energy / (exec_comm_sum + 1.0) * safety_mask
    
    # --- Key improvement 4: Fairness with bounded aging and uncertainty-triggered preemption boost ---
    wait_norm_base = ready_wait_time / (np.maximum(np.mean(ready_wait_time), eps) + eps)
    fairness_base = np.clip(wait_norm_base, 0.0, 6.0)  # extended bound for better aging resolution
    # Boost fairness only when uncertainty is high AND task is old — avoids premature preemption of fresh tasks
    fairness_boost = np.where((uncertainty > 0.5) & (ready_wait_time > np.percentile(ready_wait_time, 75) + eps),
                              uncertainty * 1.2, 0.0)
    fairness_term = fairness_base + fairness_boost
    
    # --- Key improvement 5: Robust normalization with singleton-safe minmax + fallback to zero-centering ---
    def robust_normalize(x):
        x = np.clip(x, -1000000.0, 1000000.0)
        if x.size == 1:
            return np.array([0.0])
        x_min, x_max = np.min(x), np.max(x)
        if x_max - x_min < eps:
            # All values equal → use zero-centered unit scale to preserve sign semantics
            return np.zeros_like(x) + np.sign(x[0]) * 0.5
        return (x - x_min) / (x_max - x_min + eps)
    
    norm_urgency = robust_normalize(urgency_raw)
    norm_latency = robust_normalize(latency_pressure)
    norm_energy = robust_normalize(energy_term)
    norm_fair = robust_normalize(fairness_term)
    
    # --- Key improvement 6: Adaptive coefficient tuning with violation penalty amplification ---
    # Stronger penalty for violations (now -2500), and added latency penalty weight to prevent lazy scheduling
    score = (-25.0 * norm_urgency 
             - 16.0 * norm_latency 
             - 7.0 * norm_energy 
             + 0.4 * norm_fair)
    
    # Hard violation penalty: dominates all other terms to guarantee DDL adherence priority
    violation_boost = np.where(slack < 0, -2500.0, 0.0)
    score = score + violation_boost
    
    # Final sanitization: ensure finite, deterministic, shape-(N,) output
    score = np.nan_to_num(score, nan=100000000.0, posinf=100000000.0, neginf=-100000000.0)
    score = np.clip(score, -100000000.0, 100000000.0)
    return score.astype(float).reshape(-1)
