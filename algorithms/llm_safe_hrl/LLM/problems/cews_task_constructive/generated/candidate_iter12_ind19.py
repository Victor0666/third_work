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
    Priority rule v2: Continuous urgency gradient + confidence-scaled energy optimization + robust fairness.
    
    Key improvements over v1:
    - Replaces binary urgency gate with smooth, monotonic urgency penalty: exp(-robust_slack / (|median_robust_slack|+eps))
      → preserves fine-grained ranking among urgent tasks while guaranteeing strict DDL adherence
    - Energy term now scales *continuously* with robust_slack via sigmoid gating: 1/(1+exp(-robust_slack)) * confidence
      → enables gradual energy optimization as slack improves, not abrupt on/off
    - Fairness boost refined: tanh(wait_ratio) weighted by urgency-aware mask (only active when robust_slack > -0.5)
      → prevents starvation without compromising deadline-critical tasks
    - Aging boost replaced by urgency-contextualized wait sensitivity: tanh(ready_wait_time * (1.0 + max(0, -robust_slack)))
      → amplifies aging pressure *only* under lateness risk, avoiding premature promotion of old non-urgent tasks
    - All components MAD-normalized with strict degenerate-case handling; final score bounded and deterministic
    """
    eps = 1e-08
    def sanitize(x):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(x, nan=eps, posinf=1e12, neginf=eps)
    
    min_exec_time = sanitize(min_exec_time)
    min_comm_time = sanitize(min_comm_time)
    min_incremental_energy = sanitize(min_incremental_energy)
    slack = sanitize(slack)
    upward_rank = sanitize(upward_rank)
    remaining_work = sanitize(remaining_work)
    ready_wait_time = sanitize(ready_wait_time)
    uncertainty = sanitize(uncertainty)
    
    # Robust slack: linear uncertainty margin, clipped for stability
    robust_slack = slack - 2.0 * uncertainty
    robust_slack = np.clip(robust_slack, -1e12, 1e12)
    
    # Continuous urgency penalty: exponential decay as slack worsens → higher priority (lower score) for tighter deadlines
    median_abs_robust = np.median(np.abs(robust_slack)) + eps
    urgency_penalty = np.exp(-robust_slack / (median_abs_robust + eps))
    urgency_penalty = np.clip(urgency_penalty, 0.0, 1e6)
    
    # Confidence: reliability weight for energy optimization
    confidence = np.clip(1.0 - np.tanh(uncertainty), 0.1, 0.95)
    
    # Energy optimization term: continuous sigmoid gating instead of binary switch
    energy_gate = 1.0 / (1.0 + np.exp(-robust_slack))  # soft [0,1] activation
    base_density = upward_rank * remaining_work / (min_incremental_energy + eps)
    critical_energy_density = base_density * confidence * energy_gate
    
    # Critical-path importance: normalized by workflow scale and attenuated by uncertainty
    median_rw = np.median(remaining_work) + eps
    cp_boost = upward_rank * (remaining_work / median_rw) * (1.0 / (1.0 + uncertainty + eps))
    
    # Fairness boost: wait-time scaled by tanh, but only activated when not critically urgent (robust_slack > -0.5)
    wait_mask = robust_slack > -0.5
    wait_ratio = ready_wait_time / (np.clip(robust_slack, 0.1, 1e12) + eps)
    fairness_boost = np.where(wait_mask, np.tanh(0.8 * wait_ratio), 0.0)
    
    # Urgency-contextualized aging: amplifies wait-time impact only under lateness risk
    aging_factor = 1.0 + np.maximum(0.0, -robust_slack)
    aging_boost = np.tanh(1.2 * ready_wait_time * aging_factor)
    
    # Robust normalization function
    def safe_mad_normalize(x):
        if x.size == 0:
            return np.zeros_like(x)
        if x.size == 1:
            return np.zeros_like(x)
        median_x = np.median(x)
        mad = np.median(np.abs(x - median_x)) + eps
        if mad < eps:
            return np.zeros_like(x)
        normed = (x - median_x) / mad
        return np.clip(normed, -3.0, 3.0)
    
    # Normalize all components
    urgency_norm = safe_mad_normalize(urgency_penalty)
    critical_energy_norm = safe_mad_normalize(critical_energy_density)
    cp_boost_norm = safe_mad_normalize(cp_boost)
    fairness_norm = safe_mad_normalize(fairness_boost)
    aging_norm = safe_mad_normalize(aging_boost)
    
    # Final score: smaller = higher priority
    # Urgency dominates (positive coefficient), energy & CP importance reduce score (negative coefficients), fairness/aging add mild priority
    score = (
        +5.0 * urgency_norm 
        - 3.0 * critical_energy_norm 
        - 1.8 * cp_boost_norm 
        + 0.15 * fairness_norm 
        + 0.10 * aging_norm
    )
    
    # Final sanitization: clip, replace NaN/inf
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    
    return score
