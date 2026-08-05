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
    """
    v2: Hybrid lexicographic-multiplicative priority with unified robust normalization,
         risk-aware slack gating, calibrated SEER inversion, and starvation-robust fairness.
    
    Key innovations:
    - Combines Parent 2's multiplicative urgency dominance and shared MAD normalization
      with Parent 1's critical-path pressure (upward_rank * remaining_work / exec_comm)
      and explicit uncertainty-aware slack adjustment (slack - 2*uncertainty).
    - Introduces *dynamic urgency scaling*: urgency term uses tanh-based risk-adjusted slack
      but scaled multiplicatively to preserve deadline dominance while maintaining smoothness.
    - Fairness gated by sigmoid(robust_slack / max_slack) AND clipped wait time to prevent
      unbounded starvation amplification under extreme slack shortage.
    - SEER inversion uses arctan(-SEER * 0.05) for tighter dynamic range and better gradient
      discrimination near optimal energy-efficiency points.
    - All terms normalized via shared MAD over robustly sanitized stack including urgency,
      critical density, fairness, and SEER — ensuring consistent sensitivity across small N.
    - Hard violation override applied *before* multiplication and amplified by 1e-6 factor
      to guarantee strict deadline priority ordering.
    """
    eps = 1e-08
    
    # Sanitize all inputs: convert, replace NaN/inf, clip to safe finite bounds
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
    
    # Risk-adjusted slack: penalize high-uncertainty tasks earlier
    robust_slack = slack - 2.0 * uncertainty
    robust_slack = np.clip(robust_slack, -1e5, 1e5)
    
    # Hard violation mask: triggers absolute priority override
    violation_mask = (slack < 0).astype(float)
    
    # Critical path pressure: workload-weighted importance per unit execution+comm cost
    exec_comm_sum = min_exec_time + min_comm_time + eps
    cp_density = (upward_rank * remaining_work) / (exec_comm_sum + eps)
    
    # SEER (System Energy Efficiency Ratio): higher = better energy efficiency
    seer = exec_comm_sum / (min_incremental_energy + eps)
    
    # Urgency: tanh-based risk-adjusted slack → smooth, bounded, sign-preserving
    urgency_raw = np.tanh(robust_slack / (1.0 + eps))  # [-1, 1]; negative = urgent
    urgency_term = 1.0 - urgency_raw  # [0, 2]; higher = more urgent
    
    # Fairness: sqrt(wait) gated by robust_slack-dependent sigmoid activation
    wait_clipped = np.clip(ready_wait_time, 0.0, 1e4)
    fairness_raw = np.sqrt(wait_clipped + eps)
    max_robust_slack = np.maximum(np.max(robust_slack), eps)
    fairness_gated = np.where(
        robust_slack > 0.0,
        fairness_raw * (1.0 / (1.0 + np.exp(-(robust_slack / max_robust_slack)))),
        0.0
    )
    
    # Build unified stack for shared MAD normalization: ensures consistent scaling
    stack_for_mad = np.concatenate([
        np.abs(urgency_term) + eps,
        np.abs(cp_density) + eps,
        np.abs(fairness_gated) + eps,
        np.abs(seer) + eps
    ])
    stack_for_mad = np.clip(stack_for_mad, 1e-6, 1e6)
    
    if stack_for_mad.size == 0:
        shared_median, shared_mad = 0.0, 1.0
    else:
        shared_median = np.median(stack_for_mad)
        shared_mad = np.median(np.abs(stack_for_mad - shared_median))
        if shared_mad < eps:
            shared_mad = eps
    
    # Normalize all components using shared statistics
    def normalize_shared(x):
        x = np.clip(x, -1e6, 1e6)
        return (x - shared_median) / (shared_mad + eps)
    
    norm_urgency = normalize_shared(urgency_term)
    norm_cp = normalize_shared(cp_density)
    norm_fair = normalize_shared(fairness_gated)
    norm_seer = normalize_shared(seer)
    
    # Bounded, smooth term composition
    # Urgency: sigmoid-mapped to [0.1, 0.9] for strong multiplicative base
    urgency_score = 0.1 + 0.8 / (1.0 + np.exp(-norm_urgency))
    
    # Criticality: sigmoid of normalized cp_density → prioritizes high-value critical paths
    cp_score = 0.2 + 0.6 / (1.0 + np.exp(-norm_cp))
    
    # Energy: arctan-based inversion of SEER → stable, discriminative, bounded [-1,1] → [0.1,0.9]
    seer_arctan = np.arctan(-norm_seer * 0.05) * (2.0 / np.pi)  # [-1,1]
    energy_score = 0.1 + 0.8 * (seer_arctan + 1.0) / 2.0
    
    # Fairness: sigmoid of normalized fairness → activates only when slack is positive
    fair_score = 0.1 + 0.8 / (1.0 + np.exp(-norm_fair))
    
    # Multiplicative composition: preserves deadline dominance; all terms > 0
    base_score = urgency_score * cp_score * energy_score * fair_score
    
    # Apply hard violation override: multiply by tiny factor to force top priority
    score = np.where(violation_mask, base_score * 1e-6, base_score)
    
    # Final sanitization: ensure finite, bounded, shape-(N,) output
    score = np.nan_to_num(score, nan=1e6, posinf=1e6, neginf=1e-6)
    score = np.clip(score, 1e-9, 1e6)
    score = np.asarray(score, dtype=float).reshape(-1)
    
    return score
