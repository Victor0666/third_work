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
    
    # === HARD DEADLINE ENFORCEMENT FIRST ===
    # Prioritize tasks with negative or near-zero slack *unconditionally* — no risk smoothing
    # This fixes over-smoothing: urgency must spike *immediately* when slack <= 0
    urgency_hard = np.where(slack <= 0, 1.0 + np.abs(slack) / (eps + np.maximum(0.1, -slack)), 0.0)
    
    # === ADAPTIVE TIGHTNESS FOR SOFT DEADLINES ===
    # Use robust scale: median + IQR of *positive* slack only for soft-deadline scaling
    positive_slack = np.where(slack > 0, slack, np.nan)
    med_pos_slack = np.nanmedian(positive_slack)
    med_pos_slack = np.where(np.isnan(med_pos_slack), 1.0, med_pos_slack) + eps
    q1_pos = np.nanpercentile(positive_slack, 25) if np.count_nonzero(~np.isnan(positive_slack)) > 1 else med_pos_slack * 0.5
    q3_pos = np.nanpercentile(positive_slack, 75) if np.count_nonzero(~np.isnan(positive_slack)) > 1 else med_pos_slack * 1.5
    iqr_pos = q3_pos - q1_pos + eps
    tau_soft = np.clip(med_pos_slack + 0.5 * iqr_pos, 0.1, 100.0)
    
    # Soft urgency: sigmoid centered at tau_soft, steepened for responsiveness
    urgency_soft = 1.0 / (1.0 + np.exp(2.0 * (slack - tau_soft) / (iqr_pos + eps)))
    
    # Combine: hard urgency dominates; soft adds graded priority for tight-but-positive slack
    urgency_raw = np.maximum(urgency_hard, urgency_soft)
    
    # === CRITICAL PATH TERM — PRESERVE IMPORTANCE, GATE BY SLACK RISK ===
    cp_score = upward_rank * remaining_work
    cp_med = np.median(cp_score) + eps
    cp_iqr = np.percentile(cp_score, 75) - np.percentile(cp_score, 25) + eps if cp_score.size > 1 else cp_med * 0.1
    cp_norm = (cp_score - cp_med) / (cp_iqr + eps)
    cp_scaled = np.clip(cp_norm, -3.0, 3.0) / 6.0 + 0.5
    # Gate strongly on slack < tau_soft → higher CP weight when deadline pressure rises
    cp_gate = np.clip((tau_soft - slack) / (tau_soft + eps), 0.0, 1.0)
    cp_term = cp_scaled * cp_gate
    
    # === ENERGY TERM — ACTIVATED EARLY, NOT LATE ===
    # Energy minimization starts *before* deadline violation — reward low-energy tasks early
    total_latency = min_exec_time + min_comm_time + eps
    energy_per_latency = min_incremental_energy / (total_latency + eps)
    # Energy gating now activates for *all* tasks, but weighted by slack headroom
    # Higher energy penalty when slack is large (safe to delay high-energy tasks)
    energy_weight = np.clip(slack / (tau_soft + eps), 0.0, 1.0)  # 0→1 as slack grows
    unc_damp = np.clip(1.0 - 0.5 * uncertainty, 0.1, 1.0)  # stronger damping for high uncertainty
    energy_gated = energy_per_latency * energy_weight * unc_damp
    energy_med = np.median(energy_gated) + eps
    energy_iqr = np.percentile(energy_gated, 75) - np.percentile(energy_gated, 25) + eps if energy_gated.size > 1 else energy_med * 0.1
    energy_scaled = np.clip((energy_gated - energy_med) / (energy_iqr + eps), -3.0, 3.0) / 6.0 + 0.5
    
    # === FAIRNESS — DECOUPLED FROM UNCERTAINTY, FOCUSED ON STARVATION ===
    # Only boost long-waiting tasks *if they are not already urgent* (avoid conflict with deadline)
    wait_ratio = np.clip(ready_wait_time / (total_latency + eps), 0.0, 30.0)
    fairness_gate = np.where((slack > 0) & (urgency_raw < 0.3), 1.0, 0.0)  # only non-urgent tasks
    fairness_term = wait_ratio * fairness_gate * 0.03  # mild boost to prevent starvation
    
    # === FINAL SCORE: URGENT > CRITICAL > ENERGY > FAIRNESS ===
    # Hard urgency gets highest weight; energy now contributes meaningfully *before* violation
    score = (
        0.70 * urgency_raw +
        0.18 * cp_term +
        0.10 * energy_scaled +
        0.02 * fairness_term
    )
    
    # Final sanitization: ensure finite, bounded, shape-(N,)
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e8, 1e8)
    return score.reshape(-1)
