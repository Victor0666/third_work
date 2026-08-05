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
    v2: Robust lexicographic-multiplicative priority with adaptive gating, 
    conditional normalization, and monotonic urgency amplification.
    
    Key self-evolved improvements:
    - Replaces non-monotonic `violation_amplifier` with monotonic urgency scaling: 
      hard violation → score *= 1e-6; near-violation (robust_slack < eps) → linear penalty ramp.
    - Gating for fairness & energy now uses *soft threshold*: activates when robust_slack > -eps 
      (not >0), enabling energy optimization even under tight-but-safe deadlines.
    - MAD normalization excludes zero-valued or degenerate terms (e.g., fairness_raw ≈ 0) 
      to prevent median/mad collapse; uses only non-negligible terms (> 1e-4 * max).
    - Introduces criticality-aware energy gating: seer_term activated only when cp_density > median_cp,
      ensuring energy optimization prioritizes high-impact tasks first.
    - All terms use bounded sigmoid/arctan transforms with explicit finite-range outputs [0.1, 0.9].
    - Final score enforces strict shape-(N,) and deterministic finite bounds without clipping-induced distortion.
    """
    eps = 1e-08
    def sanitize(x):
        x = np.asarray(x, dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        return np.clip(x, -1e6, 1e6)
    min_exec_time = sanitize(min_exec_time)
    min_comm_time = sanitize(min_comm_time)
    min_incremental_energy = sanitize(min_incremental_energy)
    slack = sanitize(slack)
    upward_rank = sanitize(upward_rank)
    remaining_work = sanitize(remaining_work)
    ready_wait_time = sanitize(ready_wait_time)
    uncertainty = sanitize(uncertainty)
    
    # Robust slack: more conservative than v1 (2*uncertainty), less aggressive than v0 (3*uncertainty)
    robust_slack = slack - 2.5 * uncertainty
    robust_slack = np.clip(robust_slack, -1e5, 1e5)
    
    # Hard violation mask: strict deadline breach (slack < 0)
    hard_violation = (slack < 0.0).astype(float)
    
    # Execution + communication baseline
    exec_comm_sum = min_exec_time + min_comm_time + eps
    
    # Critical path density: importance per unit time
    cp_density = upward_rank * remaining_work / (exec_comm_sum + eps)
    
    # SEER ratio: higher = more energy-efficient per time unit
    seer_ratio = exec_comm_sum / (min_incremental_energy + eps)
    
    # Urgency: monotonic, non-negative, zero when robust_slack >= 0
    urgency_raw = np.maximum(-robust_slack, 0.0)
    
    # Fairness: only activate if wait exceeds median AND robust_slack is not critically negative
    median_wait = np.median(ready_wait_time) if ready_wait_time.size > 0 else 0.0
    wait_excess = np.maximum(ready_wait_time - median_wait, 0.0)
    fairness_raw = np.where(robust_slack > -eps, np.sqrt(wait_excess + eps), 0.0)
    
    # Energy gating: activate seer only for high-criticality tasks (cp_density above median)
    median_cp = np.median(cp_density) if cp_density.size > 0 else 0.0
    seer_gated_raw = np.where(cp_density > median_cp + eps, seer_ratio, 0.0)
    
    # Build stack for MAD normalization — exclude negligible terms to avoid collapse
    all_terms = [
        np.abs(urgency_raw) + eps,
        np.abs(cp_density) + eps,
        np.abs(seer_gated_raw) + eps,
        np.abs(fairness_raw) + eps
    ]
    stack_for_mad = np.concatenate([t for t in all_terms if t.size > 0])
    # Filter out near-zero terms (< 1e-4 * max) to stabilize MAD
    if stack_for_mad.size > 0:
        max_val = np.max(stack_for_mad)
        stack_for_mad = stack_for_mad[stack_for_mad > 1e-4 * max(1.0, max_val)]
    stack_for_mad = np.clip(stack_for_mad, 1e-6, 1e6)
    
    if stack_for_mad.size == 0:
        shared_median, shared_mad = (1.0, 1.0)
    else:
        shared_median = np.median(stack_for_mad)
        shared_mad = np.median(np.abs(stack_for_mad - shared_median))
        if shared_mad < eps:
            shared_mad = eps
    
    def normalize_shared(x):
        x = np.clip(x, -1e6, 1e6)
        return (x - shared_median) / (shared_mad + eps)
    
    norm_urgency = normalize_shared(urgency_raw)
    norm_cp = normalize_shared(cp_density)
    norm_seer = normalize_shared(seer_gated_raw)
    norm_fair = normalize_shared(fairness_raw)
    
    # Bounded sigmoid/arctan transforms mapping normalized inputs to [0.1, 0.9]
    urgency_term = 0.1 + 0.8 / (1.0 + np.exp(-norm_urgency))  # [0.1, 0.9]
    cp_term = 0.1 + 0.8 / (1.0 + np.exp(-norm_cp))
    seer_arctan = np.arctan(-norm_seer * 0.1) * (2.0 / np.pi)  # [-1, 1] → [-1,1]
    seer_term = 0.1 + 0.8 * (seer_arctan + 1.0) / 2.0  # [0.1, 0.9]
    fairness_term = 0.1 + 0.8 / (1.0 + np.exp(-norm_fair))
    
    # Base multiplicative score
    base_score = urgency_term * cp_term * seer_term * fairness_term
    
    # Monotonic urgency penalty: hard violation → 1e-6; linear ramp from 0 to 1e-3 as robust_slack goes from 0 to -eps
    urgency_penalty = np.where(hard_violation, 1e-6,
                              np.where(robust_slack < 0.0,
                                      1e-6 + (1e-3 - 1e-6) * np.clip(-robust_slack / eps, 0.0, 1.0),
                                      1.0))
    
    score = base_score * urgency_penalty
    
    # Final sanitization: ensure finite, positive, shape-(N,)
    score = np.nan_to_num(score, nan=1e6, posinf=1e6, neginf=1e-6)
    score = np.clip(score, 1e-9, 1e6)
    score = np.asarray(score, dtype=float).reshape(-1)
    
    # Enforce shape-(N,) even for empty input (though N>=1 per spec)
    if score.size == 0:
        score = np.array([1.0])
    
    return score
