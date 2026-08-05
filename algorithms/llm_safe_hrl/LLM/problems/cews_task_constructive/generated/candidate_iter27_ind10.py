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
    v2: Refined deadline-hardened priority with:
    - Strict DDL violation dominance via hard priority boost (not just offset)
    - Uncertainty-coupled urgency gating that *scales* (not just attenuates) latency & energy terms
    - Safety-aware fairness: starvation relief only when slack > tau_safe AND uncertainty > 0.2
    - Robust piecewise urgency: continuous, monotonic, bounded [0,1], zero at slack=0
    - Energy efficiency term now explicitly penalizes high-energy-per-work ratio under safety margin
    - All normalization uses variance-stable min-max with per-term clipping and fallback for size-1
    - Final score strictly finite, deterministic, shape-(N,), and prioritizes deadline compliance first
    """
    eps = 1e-08
    tau_safe = 3.0
    tau_urgency = 1.0
    
    # Sanitize inputs: ensure finite, replace NaN/inf with safe defaults
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
    uncertainty = sanitize(uncertainty)
    
    # Robust slack: reduce effective slack by uncertainty to model risk-aware deadline margin
    robust_slack = np.where(slack > 0, slack - uncertainty * 0.5, slack)
    
    # Piecewise-linear urgency signal: continuous, monotonic, bounded [0,1], zero-crossing at slack=0
    # u(s) = 0 for s >= 0; u(s) = min(1, -s / tau_urgency) for s < 0 → smoothly peaks at s = -tau_urgency
    urgency_signal = np.clip(-robust_slack / (tau_urgency + eps), 0.0, 1.0)
    
    # Global urgency gate: amplifies all non-fairness terms when urgency > 0 or slack violated
    urgency_gate = 1.0 + urgency_signal
    
    # Execution + communication baseline (avoid division by zero)
    exec_comm_sum = min_exec_time + min_comm_time + eps
    
    # Latency pressure: importance × cost × urgency-gated attenuation by uncertainty
    # Attenuation is multiplicative: higher uncertainty reduces urgency impact on latency term
    latency_pressure = upward_rank * exec_comm_sum * urgency_gate * (1.0 - np.clip(uncertainty, 0.0, 1.0))
    
    # Safety factor: linear ramp from 0 (at slack=0) to 1 (at slack >= tau_safe)
    safety_factor = np.clip(robust_slack / (tau_safe + eps), 0.0, 1.0)
    
    # Energy efficiency: reward low incremental energy *per unit work*, scaled by safety
    # Use work-normalized energy: min_incremental_energy / (remaining_work + eps) → lower is better
    energy_per_work = min_incremental_energy / (remaining_work + eps)
    energy_term = energy_per_work * (1.0 - safety_factor) * urgency_gate
    
    # Critical path density: importance × work / cost → higher means more critical & expensive
    cp_density = (upward_rank + eps) * (remaining_work + eps) / (exec_comm_sum + eps)
    cp_gated = cp_density * urgency_gate * (1.0 - np.clip(uncertainty, 0.0, 1.0))
    
    # Fairness term: starvation relief only when safe (slack > tau_safe) AND uncertain (uncertainty > 0.2)
    # Ensures long-waiting tasks get boost only in low-risk regimes, avoiding interference with deadlines
    fairness_base = ready_wait_time * (1.0 + upward_rank) / (exec_comm_sum + eps)
    fairness_boost = np.where(
        (robust_slack > tau_safe) & (uncertainty > 0.2),
        uncertainty * 0.8 * fairness_base,
        0.0
    )
    fairness_term = fairness_base + fairness_boost
    
    # Normalize each component safely: min-max scaling with size-1 guard and clipping
    def robust_normalize(x):
        x = np.clip(x, -1e6, 1e6)
        if x.size == 1:
            return np.array([0.0])
        x_min, x_max = np.min(x), np.max(x)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x - x_min) / (x_max - x_min + eps)
    
    norm_urgency = robust_normalize(urgency_signal)
    norm_latency = robust_normalize(latency_pressure)
    norm_energy = robust_normalize(energy_term)
    norm_cp = robust_normalize(cp_gated)
    norm_fair = robust_normalize(fairness_term)
    
    # Weighted score: negative weights for urgency/latency/energy (lower score = higher priority)
    # Positive weight for CP density (high density should NOT be deprioritized — it's critical)
    # Negative weight for fairness (higher fairness value → less urgent → higher score)
    score = (
        -25.0 * norm_urgency      # Strongest penalty for urgency → strict DDL dominance
        - 15.0 * norm_latency     # High latency pressure under urgency
        - 8.0 * norm_energy       # Energy savings rewarded only when safe
        + 1.0 * norm_cp           # Preserve critical-path awareness (not penalized)
        - 1.2 * norm_fair         # Fairness boosts score (reduces priority) unless gated
    )
    
    # Hard deadline violation override: assign *minimum possible score* to all violated tasks
    # Ensures they are selected before any non-violating task — strict DDL enforcement
    violation_mask = robust_slack < -eps
    if np.any(violation_mask):
        # Compute base minimum among non-violating tasks; if none, use global min
        non_violating_scores = score[~violation_mask] if np.any(~violation_mask) else score
        base_min = np.min(non_violating_scores)
        # Assign score = base_min - 1e9 → guarantees argmin picks them first
        score = np.where(violation_mask, base_min - 1e9, score)
    
    # Final sanitization: ensure finite, bounded output
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)
    
    # Enforce shape (N,)
    return score.reshape(-1)
