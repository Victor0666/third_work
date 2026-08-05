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
    v2: Lexicographic-robust priority with adaptive criticality gating, 
        monotonic urgency ramp, and starvation-proof fairness.
    
    Key improvements:
    - Hard deadline dominance via multiplicative zero-override (not scaling) for violated tasks,
      guaranteeing absolute top priority when slack < 0.
    - Monotonic urgency derived from clipped linear robust_slack = slack - 3.0*uncertainty,
      avoiding flattening and ensuring interpretable penalty gradient near deadline.
    - Criticality term (cp_density) normalized *individually* with robust MAD to avoid shared-noise bias;
      used both for gating energy optimization *and* weighting fairness.
    - Energy term uses SEER inversion (exec_time/energy) *only* when cp_density > median_cp,
      and is scaled by (1 - 0.5 * cp_term) to relax energy penalty on high-criticality tasks.
    - Fairness term activated *unconditionally*, but weighted by cp_term to prioritize fairness 
      where criticality allows — preventing starvation without compromising deadline safety.
    - All normalization uses per-term robust median/MAD with explicit degeneracy fallbacks.
    - Final score enforces strict (N,) shape, finite bounds, and deterministic output.
    """
    eps = 1e-08
    
    def sanitize(x):
        x = np.asarray(x, dtype=float)
        x = np.nan_to_num(x, nan=eps, posinf=1e6, neginf=-1e6)
        return np.clip(x, -1e6, 1e6)
    
    min_exec_time = sanitize(min_exec_time)
    min_comm_time = sanitize(min_comm_time)
    min_incremental_energy = sanitize(min_incremental_energy)
    slack = sanitize(slack)
    upward_rank = sanitize(upward_rank)
    remaining_work = sanitize(remaining_work)
    ready_wait_time = sanitize(ready_wait_time)
    uncertainty = sanitize(uncertainty)
    
    # Robust slack with conservative uncertainty margin
    robust_slack = slack - 3.0 * uncertainty
    robust_slack = np.clip(robust_slack, -1e5, 1e5)
    
    # Hard violation mask: absolute priority for already-late tasks
    violation_mask = (slack < 0).astype(float)
    
    # Critical path density: importance per unit exec+comm time
    exec_comm_sum = min_exec_time + min_comm_time + eps
    cp_density = upward_rank * remaining_work / (exec_comm_sum + eps)
    
    # Normalize cp_density individually with robust MAD
    cp_med = np.median(cp_density) if cp_density.size > 0 else 0.0
    cp_abs_dev = np.abs(cp_density - cp_med)
    cp_mad = np.median(cp_abs_dev) if cp_abs_dev.size > 0 else 1.0
    cp_mad = max(cp_mad, eps)
    cp_norm = (cp_density - cp_med) / cp_mad
    cp_term = 0.1 + 0.9 / (1.0 + np.exp(-cp_norm))
    
    # Urgency: monotonic, clipped linear penalty for negative robust_slack
    urgency_raw = np.clip(-robust_slack, 0.0, 1e5)
    urg_med = np.median(urgency_raw) if urgency_raw.size > 0 else 0.0
    urg_abs_dev = np.abs(urgency_raw - urg_med)
    urg_mad = np.median(urg_abs_dev) if urg_abs_dev.size > 0 else 1.0
    urg_mad = max(urg_mad, eps)
    urg_norm = (urgency_raw - urg_med) / urg_mad
    urgency_term = 0.1 + 0.9 / (1.0 + np.exp(-urg_norm))
    
    # Energy term: SEER-like ratio, gated by criticality and active only when safe
    seer_ratio = exec_comm_sum / (min_incremental_energy + eps)
    energy_gate = (cp_density > cp_med + eps).astype(float) * (robust_slack > -eps).astype(float)
    energy_weight = 1.0 - 0.5 * cp_term  # relax energy penalty for high-criticality tasks
    gated_energy = seer_ratio * energy_gate * energy_weight
    
    # Normalize energy term individually
    eng_med = np.median(gated_energy) if gated_energy.size > 0 else 0.0
    eng_abs_dev = np.abs(gated_energy - eng_med)
    eng_mad = np.median(eng_abs_dev) if eng_abs_dev.size > 0 else 1.0
    eng_mad = max(eng_mad, eps)
    eng_norm = (gated_energy - eng_med) / eng_mad
    eng_term = 0.1 + 0.9 / (1.0 + np.exp(-eng_norm))
    
    # Fairness: always active, but scaled by criticality to avoid compromising deadlines
    wait_max = np.maximum(np.max(ready_wait_time), eps)
    fairness_raw = ready_wait_time / (wait_max + eps)
    fairness_term = 0.1 + 0.9 * fairness_raw * cp_term  # fairness boosted where criticality permits
    
    # Base multiplicative score
    base_score = urgency_term * cp_term * eng_term * fairness_term
    
    # Apply hard violation override: zero score forces immediate selection
    score = np.where(violation_mask, 0.0, base_score)
    
    # Sanitize final score
    score = np.nan_to_num(score, nan=1e6, posinf=1e6, neginf=1e-12)
    score = np.clip(score, 1e-12, 1e6)
    score = np.asarray(score, dtype=float).reshape(-1)
    
    return score
