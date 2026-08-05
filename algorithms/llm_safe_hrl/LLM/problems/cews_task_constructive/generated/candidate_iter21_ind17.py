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
        return np.nan_to_num(x, nan=eps, posinf=1000000000000.0, neginf=eps)
    min_exec_time = sanitize(min_exec_time)
    min_comm_time = sanitize(min_comm_time)
    min_incremental_energy = sanitize(min_incremental_energy)
    slack = sanitize(slack)
    upward_rank = sanitize(upward_rank)
    remaining_work = sanitize(remaining_work)
    ready_wait_time = sanitize(ready_wait_time)
    uncertainty = sanitize(uncertainty)
    
    # Robust slack: account for worst-case uncertainty propagation
    robust_slack = slack - 2.0 * uncertainty
    robust_slack = np.clip(robust_slack, -1000000000000.0, 1000000000000.0)
    
    # Urgency: piecewise-linear with adaptive threshold and smoother transition
    # Uses median-based tau to adapt to current ready-set slack distribution
    tau_base = np.median(np.abs(robust_slack)) + eps
    tau_urgency = np.clip(tau_base, 0.05, 5.0)
    urgency_raw = np.where(
        robust_slack <= 0.0,
        -robust_slack * 6.5,
        np.where(
            robust_slack <= tau_urgency,
            6.5 * (1.0 - (robust_slack / (tau_urgency + eps)) ** 1.2),
            0.0
        )
    )
    
    # Energy-latency ratio: physics-aware, bounded, and gated by deadline pressure
    total_latency = min_exec_time + min_comm_time + eps
    energy_gate = np.exp(-np.maximum(0.0, -robust_slack) / (0.08 + eps))
    # Use log-scale compression to prevent outlier dominance in energy density
    elr_base = np.log1p(min_incremental_energy / (total_latency + 3.0))
    elr_masked = np.clip(elr_base * energy_gate, 0.0, 20.0)
    
    # Critical-path density: normalized by latency and enhanced with slack margin
    cp_density = (upward_rank + eps) * (remaining_work + eps) / (total_latency + eps)
    slack_margin_norm = np.clip(np.maximum(0.0, robust_slack) / (tau_urgency + eps), 0.0, 1.0)
    cp_density_scaled = cp_density * (1.0 + 0.5 * slack_margin_norm)  # stronger boost when slack is healthy
    
    # Fairness: relative wait time now scaled per-task latency baseline (not global mean)
    # Prevents starvation of long-latency tasks
    rel_wait = np.clip(ready_wait_time / (total_latency + eps), 0.0, 10.0)
    fairness_boost = np.where(
        (robust_slack > 0.05) & (rel_wait > 0.4),
        np.clip(0.18 * (rel_wait - 0.4), 0.0, 0.18),
        0.0
    )
    
    # Uncertainty risk: now penalizes high uncertainty *only* when slack is negative,
    # and uses square-root scaling for sublinear penalty growth
    unc_risk = np.where(
        (robust_slack < -0.05) & (uncertainty > 0.1),
        np.clip(uncertainty * np.sqrt(np.abs(robust_slack) + eps), 0.0, 0.7),
        0.0
    )
    
    # MAD normalization with improved edge-case handling: always returns shape (N,)
    def safe_mad_normalize(x):
        if x.size == 0:
            return np.zeros_like(x)
        if x.size == 1:
            return np.zeros_like(x)  # no variance → zero-centered
        median_x = np.median(x)
        mad = np.median(np.abs(x - median_x)) + eps
        normed = (x - median_x) / mad
        return np.clip(normed, -3.5, 3.5)  # wider clipping for better discriminability
    
    norm_urgency = safe_mad_normalize(urgency_raw)
    norm_elr = safe_mad_normalize(elr_masked)
    norm_cp = safe_mad_normalize(cp_density_scaled)
    norm_fair = safe_mad_normalize(fairness_boost)
    norm_unc = safe_mad_normalize(unc_risk)
    
    # Weight tuning: increased urgency weight for hard-DLL compliance; reduced fairness weight to avoid over-correction;
    # added small positive weight to uncertainty risk to explicitly disincentivize risky scheduling under lateness
    score = (+7.8 * norm_urgency 
             - 4.2 * norm_elr 
             - 1.7 * norm_cp 
             - 0.25 * norm_fair 
             + 0.35 * norm_unc)
    
    # Final sanitization: ensure finite output with strict shape enforcement
    score = np.clip(score, -1000000000000.0, 1000000000000.0)
    score = np.nan_to_num(score, nan=1000000000000.0, posinf=1000000000000.0, neginf=-1000000000000.0)
    return score.reshape(-1)
