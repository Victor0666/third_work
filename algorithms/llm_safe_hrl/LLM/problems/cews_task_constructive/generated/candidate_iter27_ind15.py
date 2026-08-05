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
    
    # Robust slack-based urgency: use IQR+median instead of brittle percentile thresholds
    abs_slack = np.abs(slack)
    med_slack = np.median(abs_slack) + eps
    q1 = np.percentile(abs_slack, 25) if abs_slack.size > 1 else med_slack * 0.5
    q3 = np.percentile(abs_slack, 75) if abs_slack.size > 1 else med_slack * 1.5
    iqr = q3 - q1 + eps
    # Adaptive tightness: tighter for high-variance slack (indicating deadline pressure)
    tau_tight = np.clip(med_slack * (0.5 + 0.5 * (iqr / (med_slack + eps))), 0.05, 10.0)
    tau_loose = np.clip(med_slack + iqr, 0.5, 200.0)
    
    # Linear urgency for overdue/near-deadline tasks: smooth & bounded
    urgency_linear = np.where(slack <= 0, 
                             1.0 - np.clip(slack / (tau_tight + eps), 0.0, 1.0), 
                             0.0)
    # Sigmoid urgency for positive slack: centered at tau_loose, width scaled by IQR
    urgency_sigmoid = np.where(slack > 0,
                              1.0 / (1.0 + np.exp((slack - tau_loose) / (iqr + eps))),
                              0.0)
    urgency_raw = np.clip(np.maximum(urgency_linear, urgency_sigmoid), 0.0, 1.0)
    
    # Critical path importance: robust scaling using IQR+median to prevent rank inversion on small N
    cp_score = upward_rank * remaining_work
    cp_med = np.median(cp_score) + eps
    cp_iqr = np.percentile(cp_score, 75) - np.percentile(cp_score, 25) + eps if cp_score.size > 1 else cp_med * 0.1
    cp_norm = (cp_score - cp_med) / (cp_iqr + eps)  # Z-score style, then clamp
    cp_scaled = np.clip(cp_norm, -3.0, 3.0) / 6.0 + 0.5  # map [-3,3] → [0,1]
    
    # CP gating: full weight when slack >= 0, smoothly decayed for negative slack
    cp_gate = np.clip(1.0 + (slack / (tau_tight + eps)), 0.0, 1.0)
    cp_term = cp_scaled * cp_gate
    
    # Energy efficiency term: energy per latency, gated by slack and damped by uncertainty
    total_latency = min_exec_time + min_comm_time + eps
    energy_per_latency = min_incremental_energy / (total_latency + eps)
    # Energy gating: only active when slack > 0; stronger suppression near deadline
    energy_gate = np.where(slack > 0, 
                          np.clip(1.0 - (tau_loose - slack) / (tau_loose + eps), 0.0, 1.0),
                          0.0)
    unc_damp = np.clip(1.0 - 0.5 * uncertainty, 0.2, 1.0)  # stronger uncertainty damping
    energy_gated = energy_per_latency * energy_gate * unc_damp
    # Robust normalization: avoid percentile collapse on N=1 or skewed distributions
    energy_med = np.median(energy_gated) + eps
    energy_iqr = np.percentile(energy_gated, 75) - np.percentile(energy_gated, 25) + eps if energy_gated.size > 1 else energy_med * 0.1
    energy_scaled = np.clip((energy_gated - energy_med) / (energy_iqr + eps), -3.0, 3.0) / 6.0 + 0.5
    
    # Fairness: wait-aware boost for non-urgent tasks (slack >= 0), scaled by criticality
    wait_ratio = np.clip(ready_wait_time / (total_latency + eps), 0.0, 20.0)
    fairness_boost = np.where(slack >= 0,
                             0.05 * wait_ratio * np.clip(1.0 + 0.15 * upward_rank, 1.0, 3.0),
                             0.0)
    
    # Final weighted combination: emphasize urgency most, then fairness & CP, energy least (only when safe)
    score = (0.60 * urgency_raw + 
             0.20 * cp_term + 
             0.15 * energy_scaled + 
             0.05 * fairness_boost)
    
    # Final sanitization: ensure finite, deterministic, shape-(N,) output
    score = np.nan_to_num(score, nan=1000000000.0, posinf=1000000000.0, neginf=-1000000000.0)
    score = np.clip(score, -1000000000.0, 1000000000.0)
    return score.reshape(-1)
