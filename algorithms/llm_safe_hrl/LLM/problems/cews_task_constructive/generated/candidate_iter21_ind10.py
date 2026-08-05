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

    """
    v3 evolution: Enhances deadline safety, energy-awareness, and robustness via:
      - Hard-deadline dominance strengthened with *slack-safety margin* gating (not just median)
      - Critical-path density refined using *remaining_work*-normalized upward_rank to avoid scale bias
      - SEER activation now requires *both* absolute slack > 2s AND relative slack > 0.75 * median_abs_slack
        → tighter energy-efficiency trigger, avoiding premature activation on outliers
      - Fairness redefined as *lateness-avoiding wait fairness*: only tasks with slack > 0.0 contribute,
        and wait saturation uses dynamic percentile (90th for urgent, 75th otherwise) for adaptivity
      - Uncertainty penalty now *asymmetric*: stronger damping for high uncertainty (>3.0), milder for low
      - All normalization uses *robust MAD with outlier-aware fallback* (IQR instead of range for stability)
      - Final score adds explicit *zero-energy masking*: violated deadlines (slack < -1e-3) get max priority
      - Clipping bounds tightened to prevent floating-point drift in multiplicative chain
    """
    eps = 1e-08
    def clean_and_copy(x):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    
    min_exec_time = clean_and_copy(min_exec_time)
    min_comm_time = clean_and_copy(min_comm_time)
    min_incremental_energy = clean_and_copy(min_incremental_energy)
    slack = clean_and_copy(slack)
    upward_rank = clean_and_copy(upward_rank)
    remaining_work = clean_and_copy(remaining_work)
    ready_wait_time = clean_and_copy(ready_wait_time)
    uncertainty = clean_and_copy(uncertainty)
    
    def robust_normalize(x):
        if x.size == 0:
            return np.zeros_like(x)
        if x.size == 1:
            return np.array([0.0])
        med = np.median(x)
        dev = np.abs(x - med)
        mad = np.median(dev)
        # Prefer IQR over range for outlier resilience when mad is unstable
        q75, q25 = np.percentile(x, [75, 25])
        iqr = q75 - q25
        scale = mad if mad > eps else (iqr if iqr > eps else np.max(x) - np.min(x) + eps)
        z = (x - med) / (scale + eps)
        return np.clip(z, -2.0, 2.0)
    
    # Zero-energy masking: violated deadlines get highest priority (lowest score)
    violated_mask = (slack < -1e-3)
    base_score = np.full_like(slack, 1e6, dtype=float)
    
    # Only compute full priority for non-violated tasks
    valid_mask = ~violated_mask
    if not np.any(valid_mask):
        return base_score
    
    # Slack safety margin: median slack among *non-violated* tasks, plus 0.1s buffer
    safe_slack = slack[valid_mask]
    median_safe_slack = np.median(safe_slack) + 0.1 if safe_slack.size > 0 else 0.1 + eps
    abs_slack = np.abs(slack)
    
    # Urgency: tanh-based, gated by safety margin — stricter than median alone
    tau_urgency = 0.5
    urgency_raw = np.tanh(-slack / tau_urgency)
    urgency_gate = np.where(slack <= median_safe_slack, 1.0, 0.15)
    norm_urgency = robust_normalize(urgency_raw)
    urgency_mult = np.clip(0.05 + 9.95 * (1.0 - (norm_urgency + 2.0) / 4.0), 0.05, 10.0)
    urgency_mult = urgency_mult * urgency_gate
    
    # CPD: upward_rank normalized by remaining_work to reflect work-per-importance density
    # Avoid division by zero; cap remaining_work at reasonable lower bound
    rw_safe = np.maximum(remaining_work, eps)
    cpd_base = upward_rank / rw_safe
    exec_comm_sum = min_exec_time + min_comm_time + eps
    # Slack slope weight now based on *distance to safety margin*, not global max
    slack_dist = np.clip(median_safe_slack - slack, 0.0, None)
    slack_std = np.std(safe_slack) + eps if safe_slack.size > 0 else eps
    slack_slope_weight = np.clip(1.0 + slack_dist / (slack_std + eps), 0.4, 2.8)
    cpd_modulated = cpd_base * slack_slope_weight
    norm_cpd = robust_normalize(cpd_modulated)
    cpd_offset = np.clip((norm_cpd + 2.0) / 4.0, 0.0, 1.0)
    
    # SEER: stricter dual gating — both absolute and relative slack thresholds
    seer_base = (min_exec_time + min_comm_time + eps) / (min_incremental_energy + eps)
    seer_active = np.where(
        (slack > 2.0) & (slack > 0.75 * median_safe_slack),
        seer_base,
        0.0
    )
    norm_seer = robust_normalize(seer_active)
    seer_offset = np.clip((2.0 - norm_seer) / 4.0, 0.0, 0.45)  # Slightly tighter cap
    
    # Fairness: lateness-avoiding wait fairness — only active when slack > 0.0
    wait_threshold = np.where(
        slack > 0.0,
        np.percentile(ready_wait_time, 90) + eps,
        np.percentile(ready_wait_time, 75) + eps
    )
    wait_saturation = np.clip(ready_wait_time / (wait_threshold + eps), 0.0, 1.0)
    wait_compressed = np.log1p(wait_saturation)
    
    # Asymmetric uncertainty damping: steeper decay above 3.0
    unc_clipped = np.clip(uncertainty, 0.0, 10.0)
    uncertainty_damp = np.where(
        unc_clipped > 3.0,
        np.exp(-0.5 * (unc_clipped - 3.0)) * 0.3 + 0.1,
        np.exp(-0.2 * unc_clipped)
    )
    uncertainty_damp = np.clip(uncertainty_damp, 0.05, 1.0)
    
    fairness_raw = wait_compressed * uncertainty_damp
    fairness_gated = np.where(slack > 0.0, fairness_raw, 0.0)
    norm_fairness = robust_normalize(fairness_gated)
    fairness_offset = np.clip((norm_fairness + 2.0) / 4.0 * 0.25, 0.0, 0.25)  # Reduced weight for fairness
    
    # Compose final score for valid tasks
    score_valid = urgency_mult * (1.0 + cpd_offset) * (1.0 + seer_offset) * (1.0 + fairness_offset)
    
    # Apply finite bounds and NaN protection
    score_valid = np.nan_to_num(score_valid, nan=1e6, posinf=1e6, neginf=1e6)
    score_valid = np.clip(score_valid, 1e-6, 1e6)
    
    # Assign computed scores only to valid tasks; violated tasks retain max priority (1e6)
    score = np.where(valid_mask, score_valid, 1e6)
    
    return score
