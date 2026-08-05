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
        return np.nan_to_num(x, nan=eps, posinf=1000000.0, neginf=-1000000.0)
    min_exec_time = sanitize(min_exec_time)
    min_comm_time = sanitize(min_comm_time)
    min_incremental_energy = sanitize(min_incremental_energy)
    slack = sanitize(slack)
    upward_rank = sanitize(upward_rank)
    remaining_work = sanitize(remaining_work)
    ready_wait_time = sanitize(ready_wait_time)
    uncertainty = sanitize(uncertainty)
    
    # Total latency baseline for normalization and fairness scaling
    total_latency = min_exec_time + min_comm_time + eps
    
    # --- Urgency refinement: strict hard-deadline enforcement ---
    # Use adaptive tau_tight only for truly critical tasks (slack < 0), 
    # avoid median on empty subset by fallback to safe small value
    abs_slack = np.abs(slack)
    tight_mask = abs_slack < 1.0
    tau_tight = np.median(abs_slack[tight_mask]) if np.any(tight_mask) else 0.5
    tau_loose = np.maximum(np.median(abs_slack) + eps, 1.0)
    # Critical urgency: exponential penalty for negative slack (hard deadline violation risk)
    urgency_critical = np.where(slack < 0, np.exp(-slack / (tau_tight + eps)), 0.0)
    # Soft urgency: sigmoid centered at slack=0 with width scaled by tau_loose, ensuring monotonic decay
    urgency_soft = 1.0 / (1.0 + np.exp((slack) / (tau_loose + eps)))
    # Combine: critical dominates; soft provides gradient for positive slack
    urgency_raw = np.where(slack < 0, urgency_critical, urgency_soft)
    
    # --- Critical path importance: robust, precedence-aware normalization ---
    cp_score = upward_rank * remaining_work
    cp_med = np.median(cp_score) + eps
    cp_norm = cp_score / cp_med
    # Mask only for non-critical tasks (slack >= 0) to preserve priority of overdue tasks
    cp_mask = np.where(slack >= 0, 
                      np.clip(upward_rank / (np.percentile(upward_rank, 80) + eps), 0.0, 1.0),
                      1.0)
    cp_term = cp_norm * cp_mask
    
    # --- Energy efficiency: decoupled from slack gating to avoid penalizing low-energy tasks near DDL ---
    # Energy-per-latency remains primary efficiency signal
    energy_per_latency = min_incremental_energy / (total_latency + eps)
    # Uncertainty damping: stronger suppression for high uncertainty *and* low slack (risk-aware)
    unc_damp = np.clip(1.0 - 0.5 * uncertainty * np.clip(1.0 - (slack / (tau_loose + eps)), 0.0, 1.0), 0.2, 1.0)
    energy_gated = energy_per_latency * unc_damp
    
    # --- Fairness & starvation prevention: wait-time boost applied *only* when slack > 0 and task is non-critical ---
    # Prevents boosting overdue tasks that must run immediately anyway
    wait_ratio = np.clip(ready_wait_time / (total_latency + eps), 0.0, 8.0)
    fairness_boost = np.where(
        (slack > 0) & (upward_rank > np.percentile(upward_rank, 30, overwrite_input=False) + eps),
        0.04 * wait_ratio * (upward_rank / (np.percentile(upward_rank, 90) + eps)),
        0.0
    )
    
    # --- Unified scaling: robust percentile-based unit scaling per component ---
    def scale_to_unit(x):
        if x.size == 1:
            return np.array([0.5], dtype=float)
        p90 = np.percentile(x, 90) + eps
        # Cap outliers but preserve relative ordering within top 90%
        scaled = np.clip(x / p90, 0.0, 1.0)
        return scaled
    
    urgency_scaled = scale_to_unit(urgency_raw)
    cp_scaled = scale_to_unit(cp_term)
    energy_scaled = scale_to_unit(energy_gated)
    
    # --- Final weighted score: higher urgency weight for hard-DLL safety; fairness additive (not multiplicative)
    score = 0.6 * urgency_scaled + 0.25 * cp_scaled + 0.15 * energy_scaled + fairness_boost
    
    # Final sanitization: ensure finite, bounded output
    score = np.nan_to_num(score, nan=1000000000.0, posinf=1000000000.0, neginf=-1000000000.0)
    score = np.clip(score, -1000000000.0, 1000000000.0)
    
    # Enforce shape (N,) — critical for single-task case
    return score.reshape(-1)
