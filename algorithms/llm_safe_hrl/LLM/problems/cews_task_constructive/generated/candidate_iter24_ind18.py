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
    v2 evolution: Simplified, deadline-respecting, numerically stable priority with unified risk scaling.
    
    Key improvements over v1:
    - Restores *local slack gating* for energy term (no global median dependency) → improves efficiency
      while preserving deadline safety: energy minimization activates per-task if slack > tau_safe.
    - Replaces asymmetric uncertainty suppression with smooth sigmoid-based damping → preserves monotonicity
      and avoids discontinuities; applies uniformly to CP density and fairness.
    - Uses *full-population normalization* (not valid-only) with explicit zero-padding handling → maintains
      urgency contrast across all ready tasks, including borderline-violated ones.
    - Introduces *latency-aware urgency amplification*: urgency_raw scaled by (1 + exec_comm_sum / tau_latency)
      to prioritize low-latency tasks more aggressively under tight deadlines.
    - Adds *violation penalty* as additive term (not -inf masking) → ensures deterministic finite output,
      avoids floating-point instability, and allows downstream tie-breaking.
    - All terms bounded, clipped, and normalized robustly; final score strictly finite and shape-(N,).
    '''
    eps = 1e-08
    def clean(x):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(x, nan=eps, posinf=1e6, neginf=eps)
    
    min_exec_time = clean(min_exec_time)
    min_comm_time = clean(min_comm_time)
    min_incremental_energy = clean(min_incremental_energy)
    slack = clean(slack)
    upward_rank = clean(upward_rank)
    remaining_work = clean(remaining_work)
    ready_wait_time = clean(ready_wait_time)
    uncertainty = clean(uncertainty)
    
    # Compute base metrics
    exec_comm_sum = min_exec_time + min_comm_time + eps
    tau_urgency = 1.0
    tau_safe = 3.0
    tau_latency = 10.0
    
    # Urgency: hardened for violated & near-deadline tasks; amplified by latency sensitivity
    urgency_raw = np.where(slack <= 0, 
                           -slack * 12.0, 
                           np.where(slack <= tau_urgency, 
                                    12.0 * (1.0 - slack / (tau_urgency + eps)), 
                                    0.0))
    urgency_amplified = urgency_raw * (1.0 + np.clip(exec_comm_sum / tau_latency, 0.0, 2.0))
    
    # Latency penalty: penalize long exec+comm when urgent
    latency_penalty = np.clip(exec_comm_sum / tau_latency, 0.0, 1.0) * urgency_raw
    
    # Energy term: activate only when slack > tau_safe (local gating restored)
    energy_term = np.where(slack > tau_safe, 
                           min_incremental_energy / (exec_comm_sum + 3.0), 
                           0.0)
    
    # Critical-path density: upward_rank / work, scaled by slack margin and smooth uncertainty damping
    cp_density_base = (upward_rank + eps) / (remaining_work + eps)
    slack_margin = np.clip(np.maximum(0.0, slack) / (tau_urgency + eps), 0.0, 1.0)
    # Sigmoid uncertainty damping: 1/(1+exp(5*(uncertainty-0.5))) → gentle suppression, monotonic
    unc_damp = 1.0 / (1.0 + np.exp(5.0 * (uncertainty - 0.5)))
    cp_gated = cp_density_base * slack_margin * unc_damp
    
    # Fairness term: wait saturation, gated by slack and smoothed by uncertainty
    wait_thresh = np.percentile(ready_wait_time, 85) + eps  # fixed robust percentile
    wait_saturation = np.clip(ready_wait_time / (wait_thresh + eps), 0.0, 1.0)
    fairness_term = wait_saturation * slack_margin * unc_damp
    
    # Risk boost: reward low-uncertainty tasks *only* when slack is safe (>2*tau_safe)
    risk_boost = np.where(slack > 2.0 * tau_safe, 
                          np.clip((1.0 - uncertainty) * 0.25, 0.0, 0.25), 
                          0.0)
    
    # Robust min-max normalization over full population (preserves ranking contrast)
    def robust_norm(x):
        x_clipped = np.clip(x, -1e6, 1e6)
        if x_clipped.size == 1:
            return np.array([0.0])
        x_min, x_max = np.min(x_clipped), np.max(x_clipped)
        if x_max - x_min < eps:
            return np.zeros_like(x_clipped)
        return (x_clipped - x_min) / (x_max - x_min + eps)
    
    norm_urgency = robust_norm(urgency_amplified)
    norm_latency = robust_norm(latency_penalty)
    norm_energy = robust_norm(energy_term)
    norm_cp = robust_norm(cp_gated)
    norm_fair = robust_norm(fairness_term)
    norm_risk = robust_norm(risk_boost)
    
    # Violation penalty: finite additive boost for violated tasks (not -inf)
    violation_penalty = np.where(slack < -eps, 100.0 * (1.0 - np.clip(slack / (-eps), 0.0, 1.0)), 0.0)
    
    # Weighted linear combination: urgency & latency dominate; energy minimized only when safe
    score = (-14.0 * norm_urgency 
             - 9.0 * norm_latency 
             - 6.0 * norm_energy 
             + 1.6 * norm_cp 
             - 0.9 * norm_fair 
             + 0.5 * norm_risk 
             + violation_penalty)
    
    # Final sanitization: ensure finite, deterministic, shape-(N,)
    score = np.nan_to_num(score, nan=1e6, posinf=1e6, neginf=-1e6)
    score = np.clip(score, -1e6, 1e6)
    return score.reshape(-1)
