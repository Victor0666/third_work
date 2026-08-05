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
        x = np.nan_to_num(x, nan=0.0, posinf=1000000.0, neginf=-1000000.0)
        return np.clip(x, -1000000.0, 1000000.0)
    min_exec_time = sanitize(min_exec_time)
    min_comm_time = sanitize(min_comm_time)
    min_incremental_energy = sanitize(min_incremental_energy)
    slack = sanitize(slack)
    upward_rank = sanitize(upward_rank)
    remaining_work = sanitize(remaining_work)
    ready_wait_time = sanitize(ready_wait_time)
    uncertainty = sanitize(uncertainty)
    
    # Hierarchical urgency gating: strict DDL-first semantics enforced via hard priority layering
    # Layer 1: Hard violation → absolute highest priority (score scaled to near-zero)
    hard_violation = (slack < 0.0).astype(float)
    
    # Layer 2: Robust slack with adaptive uncertainty margin; used for urgency & gating thresholds
    robust_slack = slack - 2.5 * uncertainty
    robust_slack = np.clip(robust_slack, -100000.0, 100000.0)
    
    # Layer 3: Urgency only active when slack <= 0 or near-violation (soft deadline boundary)
    # Ensures energy fairness *never* overrides deadline safety
    urgency_active = (robust_slack <= eps).astype(float)
    
    # Critical path density: normalized by execution+comm to reflect importance per time unit
    exec_comm_sum = min_exec_time + min_comm_time + eps
    cp_density = upward_rank * remaining_work / (exec_comm_sum + eps)
    
    # SEER ratio (energy efficiency) — only considered when urgency is *not* dominant
    # i.e., only when robust_slack > eps (safe region), enabling energy optimization under DDL guarantee
    seer_ratio = exec_comm_sum / (min_incremental_energy + eps)
    seer_gated_raw = np.where(robust_slack > eps, seer_ratio, 0.0)
    
    # Fairness term: penalizes long-waiting tasks *only* in safe regime (robust_slack > 0)
    # Prevents starvation without compromising deadline adherence
    median_wait = np.median(ready_wait_time) if ready_wait_time.size > 0 else 0.0
    wait_excess = np.maximum(ready_wait_time - median_wait, 0.0)
    fairness_raw = np.where(robust_slack > eps, np.sqrt(wait_excess + eps), 0.0)
    
    # Urgency raw: monotonic, zero-based, bounded and non-negative
    urgency_raw = np.maximum(-robust_slack, 0.0)
    
    # Unified normalization domain: include only terms that are *active* in current regime
    # Avoids diluting signal with zeros — improves MAD stability and discriminative power
    active_terms = []
    if urgency_active.any():
        active_terms.append(urgency_raw + eps)
    if (robust_slack > eps).any():
        active_terms.append(cp_density + eps)
        active_terms.append(seer_gated_raw + eps)
        active_terms.append(fairness_raw + eps)
    if not active_terms:
        active_terms = [urgency_raw + eps]  # fallback: always include urgency
    
    stack_for_mad = np.concatenate([t for t in active_terms if t.size > 0])
    
    # Relative-magnitude filtering replaced with *domain-aware* filtering:
    # retain all terms with magnitude > 1e-4 * max(active_nonzero), but never drop urgency
    if stack_for_mad.size > 0:
        nonzero_mask = np.abs(stack_for_mad) > eps
        if nonzero_mask.any():
            max_val = np.max(np.abs(stack_for_mad[nonzero_mask]))
            threshold = max(1e-06, 1e-4 * max_val)
            stack_for_mad = stack_for_mad[np.abs(stack_for_mad) > threshold]
    
    if stack_for_mad.size == 0:
        shared_median, shared_mad = (1.0, 1.0)
    else:
        stack_for_mad = np.clip(stack_for_mad, 1e-06, 1000000.0)
        shared_median = np.median(stack_for_mad)
        shared_mad = np.median(np.abs(stack_for_mad - shared_median))
        if shared_mad < eps:
            shared_mad = eps
    
    def normalize_shared(x):
        x = np.clip(x, -1000000.0, 1000000.0)
        return (x - shared_median) / (shared_mad + eps)
    
    # Normalize only terms relevant to current regime
    norm_urgency = normalize_shared(urgency_raw)
    norm_cp = normalize_shared(cp_density) if (robust_slack > eps).any() else np.zeros_like(cp_density)
    norm_seer = normalize_shared(seer_gated_raw) if (robust_slack > eps).any() else np.zeros_like(seer_gated_raw)
    norm_fair = normalize_shared(fairness_raw) if (robust_slack > eps).any() else np.zeros_like(fairness_raw)
    
    # Sigmoid-based urgency term: sharp, monotonic, bounded [0.1, 0.9]
    urgency_term = 0.1 + 0.8 / (1.0 + np.exp(-norm_urgency))
    
    # CP term only active in safe regime; otherwise neutral (1.0) to avoid interference
    cp_term = np.where(robust_slack > eps, 0.1 + 0.8 / (1.0 + np.exp(-norm_cp)), 1.0)
    
    # SEER term: arctan with stronger scaling for better sensitivity in safe regime
    seer_arctan = np.arctan(-norm_seer * 0.25) * (2.0 / np.pi)
    seer_term = np.where(robust_slack > eps, 0.1 + 0.8 * (seer_arctan + 1.0) / 2.0, 1.0)
    
    # Fairness term: same logic — only active when safe
    fairness_term = np.where(robust_slack > eps, 0.1 + 0.8 / (1.0 + np.exp(-norm_fair)), 1.0)
    
    # Base score: multiplicative combination, but urgency dominates unconditionally
    base_score = urgency_term * cp_term * seer_term * fairness_term
    
    # Hard violation penalty: forces score toward minimal value (1e-06) for any violating task
    # Near-violation ramp: linear penalty from 1.0 at robust_slack=0 down to 1e-06 at robust_slack=-1.0
    # Ensures smooth, monotonic urgency amplification without discontinuities
    near_violation_ramp = np.clip(1.0 - 1.0 * np.clip(-robust_slack, 0.0, 1.0), 1e-06, 1.0)
    urgency_penalty = np.where(hard_violation, 1e-06, near_violation_ramp)
    
    score = base_score * urgency_penalty
    
    # Final sanitization: ensure finite, deterministic, shape-(N,) output
    score = np.nan_to_num(score, nan=1000000.0, posinf=1000000.0, neginf=1e-06)
    score = np.clip(score, 1e-09, 1000000.0)
    score = np.asarray(score, dtype=float).reshape(-1)
    if score.size == 0:
        score = np.array([1.0])
    return score
