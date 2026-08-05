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
    v2: Hybrid priority rule combining Parent 2's robust slack semantics and deadline-energy synergy
         with Parent 1's precise violation-margin control and starvation-resilient fairness.
    
    Key innovations:
    - Robust slack = slack - 2*uncertainty (Parent 2) for linear deadline semantics, but layered
      with Parent 1's three-mode urgency (arctan/linear/exp) applied *on robust_slack* → preserves
      gradient stability while enabling sharp violation response.
    - Critical Energy Density (CED) uses latency-aware normalization (/(exec+comm+eps)) from Parent 1,
      gated by robust_slack > 0 and weighted by confidence (1-tanh(uncertainty)) from Parent 2.
    - Fairness is dual-mode: (a) tanh-scaled wait_ratio when robust_slack > 0.1 (safe aging),
      (b) exponential aging boost when robust_slack < 0.5 (urgent starvation prevention).
    - Upward rank boosted only when robust_slack > 0 AND remaining_work > eps, attenuated by uncertainty.
    - All normalizations use unified safe_mad_normalize with outlier clipping and N=1 safeguards.
    - Strict objective hierarchy: deadline (4.8) >> CED (2.65) >> CP-boost (1.45) >> fairness (0.13, 0.11).
    """
    eps = 1e-08
    # Sanitize inputs: ensure finite, replace NaN/inf with safe defaults
    def sanitize(x):
        return np.nan_to_num(x, nan=eps, posinf=1e12, neginf=eps)
    
    min_exec_time = sanitize(np.asarray(min_exec_time, dtype=float))
    min_comm_time = sanitize(np.asarray(min_comm_time, dtype=float))
    min_incremental_energy = sanitize(np.asarray(min_incremental_energy, dtype=float))
    slack = sanitize(np.asarray(slack, dtype=float))
    upward_rank = sanitize(np.asarray(upward_rank, dtype=float))
    remaining_work = sanitize(np.asarray(remaining_work, dtype=float))
    ready_wait_time = sanitize(np.asarray(ready_wait_time, dtype=float))
    uncertainty = sanitize(np.asarray(uncertainty, dtype=float))
    
    # Compute robust slack: additive uncertainty reduction (Parent 2)
    robust_slack = slack - 2.0 * uncertainty
    robust_slack = np.clip(robust_slack, -1e12, 1e12)
    
    # Three-mode urgency on robust_slack (Parent 1 logic, adapted)
    arctan_urgency = 0.5 + 1.0 / np.pi * np.arctan(np.where(robust_slack >= 0, robust_slack, 0.0) / 5.0)
    linear_violation = np.where((robust_slack < 0) & (robust_slack >= -5), 1.0 + (-robust_slack) / 5.0, 0.0)
    severe_violation = np.where(robust_slack < -5, np.exp(np.clip(-robust_slack - 5, 0.0, 20.0)), 0.0)
    deadline_risk_raw = arctan_urgency + linear_violation + severe_violation
    
    # MAD-normalize deadline risk
    def safe_mad_normalize(x):
        if x.size == 0:
            return np.zeros_like(x)
        if x.size == 1:
            return np.zeros_like(x)
        x_clipped = np.clip(x, -1e6, 1e6)
        med = np.median(x_clipped)
        mad = np.median(np.abs(x_clipped - med)) + eps
        if mad < eps:
            return np.zeros_like(x)
        normed = (x_clipped - med) / mad
        return np.clip(normed, -4.0, 4.0)
    
    deadline_score = safe_mad_normalize(deadline_risk_raw)
    
    # Critical Energy Density: latency-aware (Parent 1) + confidence-gated (Parent 2)
    confidence = np.clip(1.0 - np.tanh(uncertainty), 0.1, 0.95)
    ced_numerator = upward_rank * remaining_work + eps
    ced_denominator = (min_incremental_energy + eps) * (min_exec_time + min_comm_time + eps)
    ced_raw = np.where(robust_slack > 0, (ced_numerator / ced_denominator) * confidence, 0.0)
    ced_norm = safe_mad_normalize(ced_raw)
    
    # Critical-path boost: gated, attenuated, workflow-normalized (hybrid)
    median_rw = np.median(remaining_work) + eps
    cp_active = np.where((robust_slack > 0) & (remaining_work > eps),
                         upward_rank * (remaining_work / median_rw) * (1.0 / (1.0 + uncertainty + eps)),
                         0.0)
    cp_boost_norm = safe_mad_normalize(cp_active)
    
    # Dual-mode fairness (Parent 1's robustness + Parent 2's context)
    # Mode A: safe aging — tanh-scaled wait_ratio when margin exists
    margin = np.clip(robust_slack, 0.1, 1e12)
    wait_ratio = ready_wait_time / (margin + eps)
    fairness_safe = np.where(robust_slack > 0.1, np.tanh(0.8 * wait_ratio), 0.0)
    
    # Mode B: urgent starvation prevention — exponential boost when robust_slack is critical
    aging_boost = np.where((robust_slack < 0.5) & (ready_wait_time > 0.05),
                          np.exp(np.clip(1.2 * ready_wait_time - 0.3 * robust_slack, 0.0, 10.0)),
                          0.0)
    
    fairness_norm = safe_mad_normalize(fairness_safe)
    aging_norm = safe_mad_normalize(aging_boost)
    
    # Final weighted score: smaller = higher priority
    # Hierarchy: deadline dominates; CED and CP-boost are energy/critical-path rewards (negative weight);
    # fairness terms are small positive boosts (to raise priority of aging tasks)
    score = (
        +4.8 * deadline_score
        - 2.65 * ced_norm
        - 1.45 * cp_boost_norm
        + 0.13 * fairness_norm
        + 0.11 * aging_norm
    )
    
    # Final sanitization: clip and replace any remaining non-finite values
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    
    return score
