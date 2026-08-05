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
    
    # Robust normalization: handles edge cases (size 0/1) and clips outliers
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
    
    # --- Urgency: hardened deadline risk assessment ---
    # Use clipped inverse slack with adaptive scaling to avoid explosion near zero slack
    # Replace tanh with smooth, bounded reciprocal: more discriminative for critical slack < 0.1s
    abs_slack_safe = np.abs(slack) + eps
    inv_slack = np.where(slack < 0, -1.0 / (abs_slack_safe + 0.1), 1.0 / (abs_slack_safe + 0.1))
    urgency_raw = np.clip(inv_slack, -10.0, 10.0)
    urgency_norm = robust_normalize(urgency_raw)
    # Linear mapping to [0.05, 2.0] ensures strict ordering dominance for violated tasks
    urgency_term = 0.05 + 1.95 * np.clip((urgency_raw + 10.0) / 20.0, 0.0, 1.0)
    
    # --- Critical-path density ratio (CPDR): enhanced by latency-aware gating ---
    total_latency = min_exec_time + min_comm_time + eps
    # Prioritize high-upward-rank *and* high-remaining-work tasks only when latency is not prohibitive
    cpdr_base = (upward_rank * remaining_work + eps) / (total_latency + eps)
    # Adaptive percentile threshold: tighten for tighter deadlines (lower median slack)
    median_slack = np.median(slack) if slack.size > 0 else 0.0
    cp_percentile = np.clip(70.0 - 20.0 * np.tanh(median_slack / 10.0), 50.0, 85.0)
    uprank_threshold = np.percentile(upward_rank, cp_percentile) if upward_rank.size > 1 else np.mean(upward_rank)
    cp_mask = (upward_rank >= uprank_threshold).astype(float)
    cp_gated = cpdr_base * cp_mask
    cp_norm = robust_normalize(cp_gated)
    cp_term = 0.2 + 1.6 * np.clip((cp_norm + 3.0) / 6.0, 0.0, 1.0)
    
    # --- Energy efficiency per latency (SEER): strengthened fairness & uncertainty damping ---
    seer_base = total_latency / (min_incremental_energy + eps)
    # Slack-gated SEER: full suppression only under violation; mild reduction for tight slack
    seer_gate = np.where(slack < 0, 0.15, 
                        np.where(slack < 1.0, 0.7 + 0.3 * (slack / 1.0), 1.0))
    # Uncertainty damping now proportional to relative uncertainty and slack margin
    avg_unc = np.mean(uncertainty + eps)
    rel_unc = uncertainty / (avg_unc + eps)
    slack_margin_ratio = np.clip((slack + 1.0) / (np.abs(np.median(slack)) + 1.0 + eps), 0.0, 2.0)
    unc_damp = np.clip(1.0 - 0.5 * rel_unc * (1.0 / (slack_margin_ratio + eps)), 0.2, 1.0)
    seer_gated = seer_base * seer_gate * unc_damp
    seer_norm = robust_normalize(seer_gated)
    seer_term = 0.3 + 1.4 * np.clip((seer_norm + 3.0) / 6.0, 0.0, 1.0)
    
    # --- Lateness fairness boost: refined with wait-time saturation and slack penalty ---
    lateness_margin = np.maximum(-slack + eps, 0.0)
    wait_ratio = ready_wait_time / (total_latency + eps)
    # Saturate fairness gain early to prevent over-prioritization of ancient ready tasks
    fairness_boost = np.where(slack < 0,
                            np.clip(0.12 * wait_ratio * np.sqrt(lateness_margin), 0.0, 0.15),
                            0.0)
    
    # --- Uncertainty risk term: redefined as additive priority penalty (not multiplicative) ---
    # Directly penalizes high uncertainty *only when slack is non-positive*, avoiding bias in safe region
    unc_risk = np.where(slack <= 0, uncertainty * (1.0 + np.tanh(-slack / 0.5)), 0.0)
    unc_norm = robust_normalize(unc_risk)
    # Convert normalized risk to additive penalty: higher risk → higher score → lower priority
    unc_penalty = 0.08 * np.clip(unc_norm, 0.0, 5.0)
    
    # --- Final score: multiplicative urgency × CPDR × SEER + additive fairness + penalty ---
    # Ensures deadline urgency dominates while preserving energy-efficiency trade-off under safety
    base_score = urgency_term * cp_term * seer_term
    score = base_score + fairness_boost + unc_penalty
    
    # Final sanitization: ensure finite, bounded output for all edge cases
    score = np.nan_to_num(score, nan=1000000000.0, posinf=1000000000.0, neginf=-1000000000.0)
    score = np.clip(score, -1000000000.0, 1000000000.0)
    
    # Enforce shape (N,) — critical for single-task case
    return score.reshape(-1)
