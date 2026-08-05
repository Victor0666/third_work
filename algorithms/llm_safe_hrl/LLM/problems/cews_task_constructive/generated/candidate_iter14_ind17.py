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
    v2: Lexicographic safety-first priority with robust slack-aware gating,
         uncertainty-amplified urgency, normalized fairness, and CED/CP synergy.
    
    Key innovations:
    - Hard violation override (slack < 0) → absolute top priority (min score)
    - Dual-threshold feasibility: energy/CPEE terms only active when slack >= median_slack_pos
    - Robust_slack = slack - 2*uncertainty used *only for gating and amplification*, not raw urgency
    - Urgency derived from tanh(robust_slack / (|median_robust_slack|+eps)) → smooth, bounded, monotonic
    - Critical Energy Density (CED) replaces CPEE: (upward_rank * remaining_work) / (energy * latency)
      gated by robust_slack > 0 AND confidence-weighted
    - CP boost attenuated by exp(-uncertainty) * I(robust_slack > 0) * (1/(1+|robust_slack|+eps))
    - Fairness = tanh(ready_wait_time / (max(robust_slack, 0.1) + eps)) → graceful aging under margin
    - All normalizations use safe MAD with strict N=1 handling, [-3.0, 3.0] clipping, and double sanitization
    - Final score clipped and nan_to_num'd to guarantee finite deterministic output.
    """
    eps = 1e-08
    
    # Sanitize all inputs: convert to float, replace NaN/inf with safe finite values
    def sanitize(x):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(x, nan=eps, posinf=1e9, neginf=1e-9)
    
    min_exec_time = sanitize(min_exec_time)
    min_comm_time = sanitize(min_comm_time)
    min_incremental_energy = sanitize(min_incremental_energy)
    slack = sanitize(slack)
    upward_rank = sanitize(upward_rank)
    remaining_work = sanitize(remaining_work)
    ready_wait_time = sanitize(ready_wait_time)
    uncertainty = sanitize(uncertainty)
    
    # Compute robust_slack for gating and amplification (not raw urgency)
    robust_slack = slack - 2.0 * uncertainty
    robust_slack = np.clip(robust_slack, -1e9, 1e9)
    
    # === HARD VIOLATION OVERRIDE ===
    violation_mask = slack < 0
    # Assign extreme low score (highest priority) to violated tasks
    base_score = np.full_like(slack, 0.0)
    
    # === URGENCY: smooth, bounded, slack-driven ===
    # Use tanh mapping for stable gradient; shift so robust_slack=0 → urgency=0.5
    median_robust = np.median(robust_slack) if robust_slack.size > 0 else 0.0
    urgency_raw = 0.5 * (1.0 - np.tanh((robust_slack - median_robust) / (np.abs(median_robust) + eps)))
    urgency_raw = np.clip(urgency_raw, 0.0, 1.0)
    
    # Uncertainty amplification: only active when robust_slack <= median_robust_slack (tight or violated)
    median_robust_pos = np.median(robust_slack[robust_slack > 0]) if np.any(robust_slack > 0) else 1.0
    amplification_mask = robust_slack <= median_robust_pos + eps
    clipped_uncert = np.clip(uncertainty, 0.0, 1.0)
    amplification_factor = np.where(amplification_mask, 1.0 + 0.7 * clipped_uncert, 1.0)
    urgency = urgency_raw * amplification_factor
    
    # === FEASIBILITY GATES ===
    # Energy/CED terms only active when robust_slack > 0 (safe margin exists)
    ced_gate = (robust_slack > 0).astype(float)
    # CP boost also requires positive robust_slack
    cp_gate = ced_gate
    
    # === CRITICAL ENERGY DENSITY (CED): energy efficiency per criticality ===
    latency_cost = min_exec_time + min_comm_time + eps
    ced_numerator = upward_rank * remaining_work + eps
    ced_denominator = (min_incremental_energy + eps) * latency_cost
    ced_base = ced_numerator / ced_denominator
    # Confidence weighting: higher confidence → stronger CED signal
    confidence = np.clip(1.0 - np.tanh(uncertainty), 0.15, 0.95)
    ced_masked = ced_base * confidence * ced_gate
    # Normalize CED: safe MAD with strict N=1 guard
    def safe_mad_normalize(x):
        x_clipped = np.clip(x, -1e6, 1e6)
        if x_clipped.size == 1:
            return np.array([0.0])
        med = np.median(x_clipped)
        mad = np.median(np.abs(x_clipped - med)) + eps
        normed = (x_clipped - med) / mad
        return np.clip(normed, -3.0, 3.0)
    ced_norm = safe_mad_normalize(ced_masked)
    
    # === CRITICAL PATH BOOST (CP): importance × work × attenuation ===
    median_rw = np.median(remaining_work) + eps
    cp_boost_raw = np.where(
        cp_gate.astype(bool),
        upward_rank * (remaining_work / median_rw) * np.exp(-uncertainty) * (1.0 / (1.0 + np.abs(robust_slack) + eps)),
        0.0
    )
    cp_norm = safe_mad_normalize(cp_boost_raw)
    
    # === FAIRNESS: contextualized aging relative to safe margin ===
    # Use robust_slack margin, but floor at 0.1 to avoid division instability
    fairness_margin = np.where(robust_slack > 0, robust_slack, 0.1)
    fairness_raw = np.tanh(0.8 * ready_wait_time / (fairness_margin + eps))
    fairness_norm = safe_mad_normalize(fairness_raw)
    
    # === COMBINE TERMS WITH LEXICOGRAPHIC WEIGHTS ===
    # Urgency dominates (violations already handled separately), then CED, CP, fairness
    # Weights chosen to reflect constraint hierarchy: deadline > energy efficiency > critical path > fairness
    score = (
        +3.0 * urgency  # primary driver: urgency amplified & smoothed
        - 1.1 * ced_norm  # secondary: minimize energy per critical work
        - 0.5 * cp_norm   # tertiary: promote critical path progress
        + 0.06 * fairness_norm  # minimal tie-breaker: prevent starvation
    )
    
    # Apply hard violation override: set score to minimum possible for violated tasks
    score = np.where(violation_mask, -1e12, score)
    
    # Final sanitization: ensure finite, deterministic output
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)
    
    return score
