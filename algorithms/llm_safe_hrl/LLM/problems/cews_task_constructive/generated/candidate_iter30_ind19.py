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
        x = np.nan_to_num(x, nan=eps, posinf=1e6, neginf=-1e6)
        return np.clip(x, -1e6, 1e6)
    min_exec_time = sanitize(min_exec_time)
    min_comm_time = sanitize(min_comm_time)
    min_incremental_energy = sanitize(min_incremental_energy)
    slack = sanitize(slack)
    upward_rank = sanitize(upward_rank)
    remaining_work = sanitize(remaining_work)
    ready_wait_time = sanitize(ready_wait_time)
    uncertainty = sanitize(uncertainty)
    
    # --- Key improvement 1: Conservative robust slack (no over-penalization) ---
    # Use additive uncertainty margin only for urgency shaping — preserve raw slack sign and magnitude for gradient.
    # Avoid zeroing scores: retain signed urgency signal for recovery scheduling.
    robust_slack = slack - uncertainty  # halved margin vs v1; avoids excessive pessimism
    
    # --- Key improvement 2: Unified urgency curve with smooth transition & bounded sensitivity ---
    # Replace piecewise linear+sigmoid with single differentiable sigmoid centered at slack=0,
    # scaled by adaptive time-scale to avoid hard thresholds.
    abs_slack = np.abs(slack)
    med_slack = np.median(abs_slack) + eps
    iqr_slack = (np.percentile(abs_slack, 75) - np.percentile(abs_slack, 25) + eps) if abs_slack.size > 1 else med_slack * 0.2
    tau_scale = np.clip(med_slack + 0.5 * iqr_slack, 0.1, 100.0)  # adaptive scale, bounded
    # Sigmoid: high priority when robust_slack << 0, low priority when >> 0, smooth around zero
    urgency_raw = 1.0 / (1.0 + np.exp(robust_slack / (tau_scale + eps)))
    
    # --- Key improvement 3: Critical-path term decoupled from uncertainty inflation ---
    # Preserve HEFT intuition: importance ∝ upward_rank × remaining_work, but damp uncertainty *only* in gating
    cp_score = upward_rank * remaining_work
    cp_med = np.median(cp_score) + eps
    cp_iqr = (np.percentile(cp_score, 75) - np.percentile(cp_score, 25) + eps) if cp_score.size > 1 else cp_med * 0.1
    cp_norm = (cp_score - cp_med) / (cp_iqr + eps)
    cp_scaled = np.clip(cp_norm, -3.0, 3.0) / 6.0 + 0.5  # [0,1] normalized importance
    # Gating: only activate CP priority when slack is non-critical (avoid starving critical tasks)
    cp_gate = np.clip(1.0 - np.maximum(0.0, robust_slack) / (tau_scale + eps), 0.0, 1.0)
    cp_term = cp_scaled * cp_gate
    
    # --- Key improvement 4: Energy term redefined per *task urgency context*, not absolute efficiency ---
    # Prioritize energy savings only when slack allows; use marginal energy *per unit work* to avoid bias toward tiny tasks
    energy_per_work = min_incremental_energy / (remaining_work + eps)
    energy_med = np.median(energy_per_work) + eps
    energy_iqr = (np.percentile(energy_per_work, 75) - np.percentile(energy_per_work, 25) + eps) if energy_per_work.size > 1 else energy_med * 0.1
    energy_norm = (energy_per_work - energy_med) / (energy_iqr + eps)
    energy_scaled = np.clip(energy_norm, -3.0, 3.0) / 6.0 + 0.5  # [0,1]
    # Energy weighting activated only under positive robust_slack, and dampened by uncertainty
    energy_gate = np.where(robust_slack > 0, 
                          np.clip(robust_slack / (tau_scale + eps), 0.0, 1.0), 
                          0.0)
    unc_damp = np.clip(1.0 - 0.3 * uncertainty, 0.3, 1.0)
    energy_gated = energy_scaled * energy_gate * unc_damp
    
    # --- Key improvement 5: Fairness boost enhanced and universally applicable ---
    # Apply wait-based fairness *even under deadline pressure*, but cap its influence
    total_latency = min_exec_time + min_comm_time + eps
    wait_ratio = np.clip(ready_wait_time / (total_latency + eps), 0.0, 10.0)
    # Boost scales with both wait time and upward_rank, but bounded and always active (no slack condition)
    fairness_boost = 0.08 * wait_ratio * np.clip(1.0 + 0.05 * upward_rank, 1.0, 2.0)
    
    # --- Final score: weighted sum with tightened coefficients to emphasize urgency & CP under stress ---
    # Increased urgency weight; reduced energy weight when slack < 0; fairness kept low but always present
    score = (
        0.60 * urgency_raw + 
        0.25 * cp_term + 
        0.10 * energy_gated + 
        0.05 * fairness_boost
    )
    
    # --- Robust final sanitization: preserve ordering, prevent degeneracy ---
    score = np.nan_to_num(score, nan=1e6, posinf=1e6, neginf=-1e6)
    score = np.clip(score, -1e6, 1e6)
    score = np.asarray(score, dtype=float).reshape(-1)
    return score
