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
    v2: Hybrid risk-aware lexicographic scheduler with critical-path energy gating,
         starvation-robust fairness, and adaptive urgency smoothing.
    
    Key improvements:
    - Combines Parent 2's tanh-based risk-adjusted slack with Parent 1's penalty gating for hard deadline violations
    - Introduces *criticality-weighted energy efficiency*: energy term scaled by upward_rank only when slack > 0, avoiding over-prioritization of non-critical low-energy tasks
    - Uses robust MAD+median normalization with explicit zero-variance fallback (Parent 2) but adds bounded clipping per Parent 1's stability practice
    - Fairness term uses sqrt(wait) + uncertainty attenuation (Parent 1) but capped linearly (Parent 2) to balance responsiveness and dominance control
    - Urgency term is *lexicographically dominant* and includes explicit late-task penalty multiplier (from Parent 1) for strict DDL enforcement
    - All terms are finite-bounded and sanitized *before* combination to preserve relative ordering under extreme inputs
    """
    eps = 1e-08
    tau = 1.0
    
    # Clean and sanitize all inputs to finite values
    def clean(x):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    
    min_exec_time = clean(min_exec_time)
    min_comm_time = clean(min_comm_time)
    min_incremental_energy = clean(min_incremental_energy)
    slack = clean(slack)
    upward_rank = clean(upward_rank)
    remaining_work = clean(remaining_work)
    ready_wait_time = clean(ready_wait_time)
    uncertainty = clean(uncertainty)
    
    # === URGENCY TERM (lexicographically dominant, DDL-first) ===
    # Risk-adjusted slack: tanh((slack - 2*uncertainty)/tau) → smooth, bounded [-1,1]
    risk_adjusted_slack = slack - 2.0 * uncertainty
    urgency_raw = np.tanh(risk_adjusted_slack / (tau + eps))
    # Map to [0, 1]: 1 = urgent (negative adjusted slack), 0 = safe (large positive)
    urgency_score = (1.0 - urgency_raw) / 2.0
    # Hard penalty for already-late tasks: multiply urgency by 3x if slack < -eps
    late_mask = (slack < -eps).astype(float)
    urgency_score = urgency_score * (1.0 + 2.0 * late_mask)
    
    # === CRITICALITY TERM (activated only when urgent) ===
    critical_mask = (slack < 0.0).astype(float)
    # Normalize remaining_work relative to max to avoid scale explosion
    norm_remaining_work = remaining_work / (np.max(remaining_work + eps) + eps)
    critical_base = upward_rank * norm_remaining_work
    # Robust MAD normalization with zero-variance fallback
    def normalize_mad(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        median_val = np.median(x)
        mad = np.median(np.abs(x - median_val))
        if mad < eps:
            return np.zeros_like(x)
        normed = (x - median_val) / (mad + eps)
        return np.clip(normed, -4.0, 4.0)
    norm_critical = normalize_mad(critical_base)
    critical_term = 2.5 * norm_critical * critical_mask
    
    # === ENERGY TERM (active only when slack > 0, weighted by criticality) ===
    energy_gate = (slack > 0.0).astype(float)
    # SEER-like ratio: energy per unit latency; invert to prioritize efficiency
    seer_ratio = min_incremental_energy / (min_exec_time + min_comm_time + eps)
    energy_base = 1.0 / (seer_ratio + eps)
    # Criticality-weighted energy: only reward efficiency on high-upward-rank tasks
    energy_base_weighted = energy_base * (upward_rank / (np.max(upward_rank + eps) + eps))
    norm_energy = normalize_mad(energy_base_weighted)
    energy_term = -1.8 * norm_energy * energy_gate
    
    # === FAIRNESS TERM (sqrt wait + uncertainty attenuation, capped) ===
    max_wait = np.max(ready_wait_time + eps)
    capped_wait = np.minimum(ready_wait_time, max_wait * 0.5)
    wait_term = np.sqrt(capped_wait + eps)
    # Attenuate fairness weight for high-uncertainty tasks (they need faster dispatch)
    uncert_attenuation = np.clip(1.0 - uncertainty / (np.max(uncertainty + eps) + eps), 0.0, 1.0)
    fairness_raw = wait_term * uncert_attenuation
    norm_fairness = normalize_mad(fairness_raw)
    fairness_term = -0.3 * norm_fairness
    
    # === COMBINED SCORE (urgency dominates, then criticality, energy, fairness) ===
    score = 4.0 * urgency_score + critical_term + energy_term + fairness_term
    
    # Final sanitization: ensure finite, bounded output
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e8, 1e8)
    
    # Ensure shape (N,) — no broadcasting or scalar collapse
    return score.reshape(-1)
