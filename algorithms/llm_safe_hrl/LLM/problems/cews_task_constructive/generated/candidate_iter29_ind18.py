import numpy as np

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
    # Sanitize inputs: ensure finite, bounded, float32/64, no NaN/inf
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
    
    # Robust slack: account for uncertainty margin; clamp to avoid overflow
    robust_slack = slack - 2.5 * uncertainty
    robust_slack = np.clip(robust_slack, -1e5, 1e5)
    
    # Hard deadline violation flag (strict DDL enforcement)
    hard_violation = (slack < 0.0).astype(float)
    
    # Critical path density: importance-weighted work per latency unit
    exec_comm_sum = min_exec_time + min_comm_time + eps
    cp_density = (upward_rank * remaining_work) / (exec_comm_sum + eps)
    
    # Energy efficiency ratio: higher is better (SEER-like: work/energy)
    # Inverted for minimization: lower energy per latency → higher priority
    seer_ratio = exec_comm_sum / (min_incremental_energy + eps)
    
    # Urgency: monotonic non-negative penalty for negative robust_slack
    urgency_raw = np.maximum(-robust_slack, 0.0)
    
    # Fairness term: reward long-waiting tasks only when slack permits (soft gating)
    median_wait = np.median(ready_wait_time) if ready_wait_time.size > 0 else 0.0
    wait_excess = np.maximum(ready_wait_time - median_wait, 0.0)
    fairness_raw = np.where(robust_slack > -eps, np.sqrt(wait_excess + eps), 0.0)
    
    # Criticality-aware energy gating: only optimize energy on high-impact tasks
    median_cp = np.median(cp_density) if cp_density.size > 0 else 0.0
    seer_gated_raw = np.where(cp_density > median_cp + eps, seer_ratio, 0.0)
    
    # Build term stack for shared MAD normalization — exclude negligible values
    all_terms = [urgency_raw + eps, cp_density + eps, seer_gated_raw + eps, fairness_raw + eps]
    stack_for_mad = np.concatenate([t for t in all_terms if t.size > 0])
    
    # Filter out near-zero terms to avoid MAD collapse (keep > 0.0001 * max non-zero)
    if stack_for_mad.size > 0:
        max_val = np.max(np.abs(stack_for_mad))
        threshold = max(1e-6, 1e-4 * max_val)
        stack_for_mad = stack_for_mad[np.abs(stack_for_mad) > threshold]
    
    # Ensure stack has content; fallback to safe defaults if degenerate
    if stack_for_mad.size == 0:
        shared_median, shared_mad = (1.0, 1.0)
    else:
        stack_for_mad = np.clip(stack_for_mad, 1e-6, 1e6)
        shared_median = np.median(stack_for_mad)
        shared_mad = np.median(np.abs(stack_for_mad - shared_median))
        if shared_mad < eps:
            shared_mad = eps
    
    # Shared normalization function
    def normalize_shared(x):
        x = np.clip(x, -1e6, 1e6)
        return (x - shared_median) / (shared_mad + eps)
    
    norm_urgency = normalize_shared(urgency_raw)
    norm_cp = normalize_shared(cp_density)
    norm_seer = normalize_shared(seer_gated_raw)
    norm_fair = normalize_shared(fairness_raw)
    
    # Sigmoid-transformed terms: map normalized scores to [0.1, 0.9] for stable multiplicative composition
    urgency_term = 0.1 + 0.8 / (1.0 + np.exp(-norm_urgency))
    cp_term = 0.1 + 0.8 / (1.0 + np.exp(-norm_cp))
    
    # SEER term uses arctan for smoother saturation and monotonic inverse behavior
    seer_arctan = np.arctan(-norm_seer * 0.15) * (2.0 / np.pi)
    seer_term = 0.1 + 0.8 * (seer_arctan + 1.0) / 2.0
    
    # Fairness term: prioritize fairness under safety, but suppress when urgent
    fairness_term = 0.1 + 0.8 / (1.0 + np.exp(-norm_fair))
    
    # Base multiplicative priority: all terms contribute multiplicatively
    base_score = urgency_term * cp_term * seer_term * fairness_term
    
    # Urgency penalty: strict exponential scaling for violations, linear ramp near zero
    # Ensures hard DDL compliance first, then smooth optimization within feasibility
    urgency_penalty = np.where(
        hard_violation,
        1e-06,
        np.where(
            robust_slack < 0.0,
            1e-06 + (0.002 - 1e-06) * np.clip(-robust_slack / (eps + 1.0), 0.0, 1.0),
            1.0
        )
    )
    
    # Final score: apply penalty, sanitize, enforce shape-(N,)
    score = base_score * urgency_penalty
    score = np.nan_to_num(score, nan=1e6, posinf=1e6, neginf=1e-06)
    score = np.clip(score, 1e-09, 1e6)
    score = np.asarray(score, dtype=float).reshape(-1)
    
    # Guarantee shape (N,) even for N=0 or N=1 (avoid scalar)
    if score.size == 0:
        score = np.array([1.0])
    
    return score
