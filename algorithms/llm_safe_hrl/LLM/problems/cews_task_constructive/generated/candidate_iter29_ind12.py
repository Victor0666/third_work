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
    v2: Hybrid deadline-safe, energy-aware, and fairness-balanced priority.
    
    Key innovations:
    - Combines Parent 2's smooth arctan urgency (monotonic, no discontinuity) with Parent 1's explicit violation boost for hard-deadline guarantee.
    - Uses robust quantile normalization (Parent 2) but adds N=1 safety fallback and zero-variance handling from Parent 1.
    - Introduces 'risk-adjusted criticality': upward_rank * (1 + uncertainty) gated by slack safety, replacing raw criticality scaling.
    - Energy term uses Parent 1's efficiency quotient (energy/(exec+comm+1)) but gated multiplicatively by urgency_scaled (Parent 2 style).
    - Fairness uses log1p(wait) + linear age boost for small-N stability; gated only by slack > 0 and upward_rank > median (Parent 2), but with additive tie-break when urgency is low.
    - All operations finite, deterministic, and N=1 safe; returns shape (N,).
    """
    eps = 1e-08
    
    # Sanitize inputs: ensure finite, replace NaN/inf
    def sanitize(x):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(x, nan=eps, posinf=1e6, neginf=-1e6)
    
    min_exec_time = sanitize(min_exec_time)
    min_comm_time = sanitize(min_comm_time)
    min_incremental_energy = sanitize(min_incremental_energy)
    slack = sanitize(slack)
    upward_rank = sanitize(upward_rank)
    remaining_work = sanitize(remaining_work)
    ready_wait_time = sanitize(ready_wait_time)
    uncertainty = np.clip(sanitize(uncertainty), 0.0, 1.0)  # bound uncertainty
    
    # --- Urgency computation: smooth arctan + hard-violation boost ---
    # Robust slack: penalize uncertainty only when slack is positive (avoid over-penalizing critical tasks)
    robust_slack = np.where(slack > 0, slack - 1.5 * uncertainty, slack)
    # Smooth monotonic urgency: arctan maps slack → [0,1], higher urgency when slack is negative or near zero
    urgency_raw = np.arctan((eps - robust_slack) / (eps + 0.1))
    urgency_scaled = (np.pi / 2 - urgency_raw) / np.pi  # [0,1], 1 = most urgent
    # Hard-deadline override: explicit high-priority boost for violated tasks (ensures DDL adherence)
    violation_boost = np.where(slack < 0, -1e4, 0.0)
    
    # --- Latency pressure: critical-path aware and uncertainty-dampened ---
    exec_comm_sum = min_exec_time + min_comm_time + eps
    # Risk-adjusted criticality: upward_rank amplified by uncertainty only when slack allows
    risk_criticality = upward_rank * (1.0 + np.where(robust_slack > 0, uncertainty, 0.0))
    latency_pressure = risk_criticality * exec_comm_sum
    
    # --- Energy efficiency: efficiency quotient gated by urgency ---
    # Energy per unit work-time: lower is better; scaled only when safe (urgency < 0.95 avoids over-prioritizing low-energy trivial tasks)
    energy_quotient = min_incremental_energy / (exec_comm_sum + 1.0)
    safety_factor = np.clip(robust_slack / (2.0 + eps), 0.0, 1.0)
    energy_term = energy_quotient * safety_factor * urgency_scaled
    
    # --- Fairness: anti-starvation with dual gating and tie-breaking ---
    wait_fairness = np.log1p(ready_wait_time) + 0.1 * ready_wait_time  # smoother + linear tail
    urank_median = np.median(upward_rank) if len(upward_rank) > 1 else np.mean(upward_rank)
    # Gate fairness to non-critical tasks (slack > 0.1) AND high-impact (upward_rank > median)
    fairness_gate = np.where((robust_slack > 0.1) & (upward_rank > urank_median), 1.0, 0.0)
    # Add small unconditional tie-break for long-waiting tasks when urgency is low (prevents starvation)
    tie_break = np.where(urgency_scaled < 0.2, 0.01 * ready_wait_time, 0.0)
    fairness_term = (wait_fairness * fairness_gate + tie_break) * (1.0 - urgency_scaled)
    
    # --- Robust quantile normalization with small-N fallback ---
    def quantile_normalize(x):
        x = np.clip(x, -1e6, 1e6)
        if x.size == 1:
            return np.array([0.0])
        q10 = np.quantile(x, 0.1)
        q90 = np.quantile(x, 0.9)
        if q90 - q10 < eps:
            return np.zeros_like(x)
        return (x - q10) / (q90 - q10 + eps)
    
    norm_urgency = quantile_normalize(urgency_scaled)
    norm_latency = quantile_normalize(latency_pressure)
    norm_energy = quantile_normalize(energy_term)
    norm_fair = quantile_normalize(fairness_term)
    
    # --- Weighted score: urgency dominates, others support under safety ---
    # Negative weights for terms where lower value = higher priority (urgency, latency, energy)
    # Positive weight for fairness (higher fairness_score → lower priority, so we *add* it)
    score = (
        -25.0 * norm_urgency 
        - 12.0 * norm_latency 
        - 10.0 * norm_energy 
        + 3.0 * norm_fair
    )
    
    # Apply hard-violation boost (additive, dominates all)
    score = score + violation_boost
    
    # Final sanitization: ensure finite and bounded
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)
    
    return score.astype(float).reshape(-1)
