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
    v2: Hybrid lexicographic-multiplicative priority with robust slack gating,
    unified MAD normalization, calibrated urgency dominance, and starvation-robust fairness.
    
    Key innovations:
    - Combines Parent 2's multiplicative urgency dominance (preserving hard DDL enforcement) 
      with Parent 1's risk-adjusted slack (slack - 2*uncertainty) for better uncertainty-awareness.
    - Uses shared MAD normalization across *all* terms (urgency, criticality, energy, fairness) 
      for consistent sensitivity scaling — improves stability on small N.
    - Introduces dynamic fairness activation: only when robust_slack > 0 AND wait_time > median_wait,
      preventing premature starvation while avoiding fairness dominance during urgency.
    - Energy term uses inverted SEER ratio with arctan stabilization (Parent 2) but gated by positive robust_slack
      (Parent 1 logic), ensuring energy optimization only when deadlines are safe.
    - Hard violation override applied multiplicatively *before* composition and amplified by urgency magnitude.
    - All operations protected against zero/nan/inf; final score strictly finite and shape-(N,) enforced.
    """
    eps = 1e-08
    
    # Sanitize all inputs to finite values
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
    
    # Robust slack: account for uncertainty margin (Parent 1) + tighter bound (Parent 2)
    robust_slack = slack - 2.0 * uncertainty
    robust_slack = np.clip(robust_slack, -1e5, 1e5)
    
    # Violation mask for hard deadline enforcement
    violation_mask = (slack < 0.0).astype(float)
    
    # Compute base terms
    exec_comm_sum = min_exec_time + min_comm_time + eps
    cp_density = upward_rank * remaining_work / (exec_comm_sum + eps)  # critical path density
    seer_ratio = exec_comm_sum / (min_incremental_energy + eps)         # energy efficiency ratio
    urgency_raw = np.maximum(-robust_slack, 0.0)                        # risk-adjusted urgency
    
    # Fairness: activate only when robust_slack > 0 AND wait_time exceeds median (starvation-robust)
    median_wait = np.median(ready_wait_time) if ready_wait_time.size > 0 else 0.0
    fairness_raw = np.where(
        (robust_slack > 0.0) & (ready_wait_time > median_wait + eps),
        np.sqrt(np.maximum(ready_wait_time - median_wait, 0.0) + eps),
        0.0
    )
    
    # Build stack for shared MAD normalization across all key terms
    stack_terms = [
        np.abs(urgency_raw) + eps,
        np.abs(cp_density) + eps,
        np.abs(seer_ratio) + eps,
        np.abs(fairness_raw) + eps
    ]
    stack_for_mad = np.concatenate(stack_terms)
    stack_for_mad = np.clip(stack_for_mad, 1e-6, 1e6)
    
    if stack_for_mad.size == 0:
        shared_median, shared_mad = 0.0, 1.0
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
    norm_seer = normalize_shared(seer_ratio)
    norm_fair = normalize_shared(fairness_raw)
    
    # Sigmoid-transformed components (bounded [0.1, 0.99] for stable multiplication)
    urgency_term = 0.1 + 0.9 / (1.0 + np.exp(-norm_urgency))
    cp_term = 0.1 + 0.9 / (1.0 + np.exp(-norm_cp))
    # Energy: arctan-stabilized inverted SEER, active only when robust_slack > 0
    seer_arctan = np.arctan(-norm_seer * 0.1) * (2.0 / np.pi)
    seer_term = 0.1 + 0.9 * (seer_arctan + 1.0) / 2.0
    seer_gated = np.where(robust_slack > 0.0, seer_term, 0.1)  # baseline when unsafe
    # Fairness: activated only in safe & starved regime
    fairness_term = 0.1 + 0.9 / (1.0 + np.exp(-norm_fair))
    
    # Multiplicative base score: urgency dominates compositionally
    base_score = urgency_term * cp_term * seer_gated * fairness_term
    
    # Hard violation override: amplify urgency dominance multiplicatively
    # Use magnitude of urgency_raw to scale override strength
    violation_amplifier = np.where(
        violation_mask, 
        1e-6 * (1.0 + urgency_raw / (np.maximum(np.max(urgency_raw), eps) + eps)), 
        1.0
    )
    score = base_score * violation_amplifier
    
    # Final sanitization: ensure finite, positive, shape-(N,)
    score = np.nan_to_num(score, nan=1e6, posinf=1e6, neginf=1e-6)
    score = np.clip(score, 1e-9, 1e6)
    score = np.asarray(score, dtype=float).reshape(-1)
    
    # Enforce shape (N,) even for N=1
    if score.size == 0:
        score = np.array([1.0])
    
    return score
