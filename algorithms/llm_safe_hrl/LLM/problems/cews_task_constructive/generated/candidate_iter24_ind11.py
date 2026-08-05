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
    
    def robust_normalize(x):
        if x.size == 0:
            return np.zeros_like(x)
        if x.size == 1:
            return np.zeros_like(x)
        q25, q75 = np.percentile(x, [25, 75])
        iqr = q75 - q25 + eps
        med = np.median(x)
        normed = (x - med) / iqr
        return np.clip(normed, -5.0, 5.0)
    
    # --- Urgency: preserve gradient near deadline & avoid hard zeroing ---
    # Use smoothed sigmoid-like urgency with full dynamic range; avoid masking to retain recoverable signal
    abs_slack = np.abs(slack)
    tau_urg = np.median(abs_slack[abs_slack > eps]) if np.any(abs_slack > eps) else np.mean(abs_slack) + eps
    tau_urg = np.maximum(tau_urg, 0.1)
    # Soft urgency: higher score when slack is negative or small positive → lower priority (so invert later via scaling)
    urgency_raw = np.tanh(-slack / tau_urg)  # [-1,1]: -1→urgent, +1→leisurely
    urgency_norm = robust_normalize(urgency_raw)
    # Map to [0.05, 2.0]: higher value = higher urgency weight (not priority score)
    urgency_weight = 0.05 + 1.95 * (urgency_raw + 1.0) / 2.0
    
    # --- Critical Path Dominance Ratio (CPDR): improved stability & interpretability ---
    total_latency = min_exec_time + min_comm_time + eps
    # Use robust percentile threshold even for small N
    uprank_thresh = np.percentile(upward_rank, 75) if len(upward_rank) > 1 else np.max(upward_rank)
    cpdr_mask = (upward_rank >= uprank_thresh).astype(float)
    # Avoid division by tiny latency; prioritize high work/latency ratio on critical path
    cpdr_base = (upward_rank * remaining_work + eps) / (total_latency + eps)
    cpdr_gated = cpdr_base * cpdr_mask
    cpdr_norm = robust_normalize(cpdr_gated)
    # Normalize to [0.2, 1.8] — wider range for stronger discrimination on critical path
    cpdr_weight = 0.2 + 1.6 * np.clip((cpdr_norm + 3.0) / 6.0, 0.0, 1.0)
    
    # --- SEER (Scheduling Efficiency Energy Ratio): fixed inversion & uncertainty damping ---
    # Higher SEER = better energy-latency trade-off → should *lower* priority score (i.e., promote)
    seer_base = total_latency / (min_incremental_energy + eps)  # latency per joule → larger = worse efficiency
    # Gate strongly only under violation risk, but keep gradient: soften gate from 0.2→0.4 for slack<0
    seer_gate = np.where(slack < 0, 0.4, 1.0)
    # Uncertainty damping: reduce SEER confidence when uncertainty high, but never below 0.2
    unc_damp = np.clip(1.0 - 0.5 * (uncertainty / (np.mean(uncertainty + eps) + eps)), 0.2, 1.0)
    seer_gated = seer_base * seer_gate * unc_damp
    seer_norm = robust_normalize(seer_gated)
    # Invert correctly: high SEER (inefficient) → high term → high score → low priority; map to [0.3, 1.7]
    seer_weight = 0.3 + 1.4 * np.clip((seer_norm + 3.0) / 6.0, 0.0, 1.0)
    
    # --- Fairness & Lateness Recovery: broaden scope beyond tight deadlines ---
    # Boost fairness for *all* tasks with slack <= median_slack (not just ≤2s), scaled by wait/latency ratio and lateness depth
    median_slack = np.median(slack) if slack.size > 0 else 0.0
    fairness_mask = (slack <= median_slack + eps) & (ready_wait_time > eps)
    wait_ratio = ready_wait_time / (total_latency + eps)
    # Depth-aware boost: stronger for more negative slack, capped
    lateness_depth = np.maximum(-slack + eps, 0.0)
    fairness_boost = np.where(
        fairness_mask,
        np.clip(0.12 * wait_ratio * np.sqrt(lateness_depth), 0.0, 0.12),
        0.0
    )
    
    # --- Composite priority score: multiplicative urgency×criticality×efficiency, plus additive fairness ---
    # All weights ∈ [0.05, 2.0]; product emphasizes joint constraints; fairness breaks ties toward waiting tasks
    base_score = urgency_weight * cpdr_weight * seer_weight
    score = base_score + fairness_boost
    
    # --- Final sanitization: ensure finite, bounded, shape-(N,) output ---
    score = np.nan_to_num(score, nan=1000000.0, posinf=1000000.0, neginf=1000000.0)
    score = np.clip(score, 1e-08, 1000000.0)
    # Enforce shape (N,) explicitly — critical for N=1 case
    return score.reshape(-1)
