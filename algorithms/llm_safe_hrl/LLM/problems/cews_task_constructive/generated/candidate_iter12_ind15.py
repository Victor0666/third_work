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
    v3: Simplified, constraint-first priority rule — deadline safety is hard-gated, energy optimization is smooth and uncertainty-aware.
    
    Key improvements:
    - Replaces multi-mode urgency with single monotonic robust_slack → tanh mapping: stable gradient, interpretable, no piecewise artifacts.
    - Deadline gating is strict: energy/critical-path terms only active when robust_slack > 0 (hard feasibility guard).
    - Fairness unified as *contextualized aging*: tanh(wait_ratio) × I(robust_slack > 0.1) — no separate aging boost to avoid noise amplification.
    - Critical Energy Density (CED) uses latency-normalized denominator (exec+comm+eps) and confidence weighting — now scaled by robust_slack margin for graceful deactivation near deadline.
    - Upward rank boost attenuated by both uncertainty and robust_slack (exponential decay beyond safe margin) → avoids over-prioritizing low-risk critical tasks.
    - All normalizations use safe_mad_normalize with strict N=1/constant safeguards and conservative clipping.
    - Objective weights derived from constraint hierarchy: deadline (1.0) dominates; CED (-0.85) and CP (-0.4) are secondary; fairness (+0.08) is minimal tie-breaker.
    - No unbounded ops: all divisions, exp, tanh, arctan guarded; final score finite, clipped, and deterministic.
    """
    eps = 1e-08
    
    # Sanitize inputs: replace NaN/inf with safe finite values
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
    
    # Robust slack: linear, interpretable, uncertainty-aware
    robust_slack = slack - 2.0 * uncertainty
    robust_slack = np.clip(robust_slack, -1e12, 1e12)
    
    # Monotonic urgency: tanh maps robust_slack → [-1,1], shifted & scaled to [0,1] for intuitive semantics
    # Negative robust_slack → high urgency; large positive → low urgency
    urgency_raw = 0.5 * (1.0 - np.tanh(robust_slack / (np.abs(np.median(robust_slack)) + eps)))
    urgency = np.clip(urgency_raw, 0.0, 1.0)
    
    # Safe MAD normalization — handles N=1, constants, outliers
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
    
    urgency_norm = safe_mad_normalize(urgency)
    
    # Confidence: reliability weight for energy/critical-path terms
    confidence = np.clip(1.0 - np.tanh(uncertainty), 0.1, 0.95)
    
    # Critical Energy Density: higher density = better energy efficiency per latency cost
    # Activated only when robust_slack > 0; smoothly attenuated as robust_slack → 0+
    latency_cost = min_exec_time + min_comm_time + eps
    ced_numerator = upward_rank * remaining_work + eps
    ced_denominator = (min_incremental_energy + eps) * latency_cost
    ced_base = ced_numerator / ced_denominator
    
    # Soft gating: weight by sigmoid of robust_slack to fade out near deadline boundary
    ced_gate = 1.0 / (1.0 + np.exp(-robust_slack / (np.abs(np.median(robust_slack)) + eps) + 2.0))
    ced_raw = ced_base * confidence * ced_gate * (robust_slack > 0)
    ced_norm = safe_mad_normalize(ced_raw)
    
    # Critical Path Boost: importance × scale × uncertainty attenuation
    # Only active when robust_slack > 0 and remaining_work > eps
    median_rw = np.median(remaining_work) + eps
    cp_active = (
        (robust_slack > 0) & 
        (remaining_work > eps)
    )
    cp_boost_raw = np.where(
        cp_active,
        upward_rank * (remaining_work / median_rw) * np.exp(-uncertainty) * (1.0 / (1.0 + np.abs(robust_slack) + eps)),
        0.0
    )
    cp_boost_norm = safe_mad_normalize(cp_boost_raw)
    
    # Contextualized Fairness: aging matters only when safe (robust_slack > 0.1)
    margin = np.clip(robust_slack, 0.1, 1e12)
    wait_ratio = ready_wait_time / (margin + eps)
    fairness_raw = np.where(robust_slack > 0.1, np.tanh(0.7 * wait_ratio), 0.0)
    fairness_norm = safe_mad_normalize(fairness_raw)
    
    # Final score: deadline dominates; others are secondary corrections
    # Smaller score = higher priority → urgency_norm (0→1) is positive; others subtracted
    score = (
        +1.0 * urgency_norm
        - 0.85 * ced_norm
        - 0.4 * cp_boost_norm
        + 0.08 * fairness_norm
    )
    
    # Final sanitization: ensure finite, bounded, deterministic output
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    
    return score
