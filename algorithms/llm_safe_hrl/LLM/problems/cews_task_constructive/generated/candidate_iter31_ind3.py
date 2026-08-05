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
    v2: Robust deadline-anchored priority with IQR-based normalization,
    SEER-inspired criticality-energy coupling, uncertainty-gated fairness,
    and monotonic arctan urgency (no clipping, smooth gradient at zero slack).
    
    Key mutations:
    - Replaces piecewise-linear urgency with arctan(-slack/tau_urgency) → smooth, bounded, monotonic,
      avoids discontinuities and artificial thresholds; maps slack=0 → 0, slack→-∞ → +π/2.
    - Uses IQR-based robust scaling (Q1/Q3) instead of min-max → less sensitive to outliers in real traces.
    - Introduces 'SEER-coupling': energy term is *inverted* only when both (i) slack > tau_safe AND
      (ii) cp_density > median(cp_density), focusing savings on high-bottleneck, safe tasks.
    - Fairness now uses uncertainty-weighted waiting *only for slack > 0*, but scaled by 1/(1+uncertainty)
      to avoid over-prioritizing noisy long-waiters.
    - Removes hardcoded coefficients; weights derived from relative variance of normalized terms
      to auto-balance signal contributions.
    - Adds 'latency resilience' term: (min_exec_time + min_comm_time) * (1 + uncertainty),
      penalizing high-risk latency-sensitive tasks when slack is tight.
    - All normalization includes epsilon-guarded fallback for singleton arrays.
    """
    eps = 1e-8
    tau_safe = np.percentile(slack[slack > eps], 25) if np.any(slack > eps) else 2.0
    tau_safe = np.clip(tau_safe, 0.5, 10.0)
    tau_urgency = np.percentile(np.abs(slack), 75) if slack.size > 1 else 3.0
    tau_urgency = np.clip(tau_urgency, 0.1, 20.0)

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

    # Smooth, bounded urgency: arctan(-slack/tau) → [0, π/2]; higher = more urgent
    urgency = np.arctan(-slack / (tau_urgency + eps))

    # Robust latency resilience: penalize high-latency + high-uncertainty under pressure
    exec_comm_sum = min_exec_time + min_comm_time + eps
    latency_resilience = exec_comm_sum * (1.0 + np.clip(uncertainty, 0.0, 2.0))
    # Gate by urgency: only amplify when slack <= tau_urgency (moderately tight or worse)
    latency_resilience = np.where(slack <= tau_urgency, latency_resilience, 0.0)

    # Criticality-aware energy term: invert only when safe AND high bottleneck risk
    cp_density = (upward_rank + eps) * (remaining_work + eps) / (exec_comm_sum + eps)
    cp_median = np.median(cp_density) if cp_density.size > 1 else np.mean(cp_density)
    seer_active = (slack > tau_safe) & (cp_density > cp_median)
    # Energy priority: lower energy is better → negate when SEER active, else keep raw cost
    energy_term = np.where(seer_active, -min_incremental_energy, min_incremental_energy)

    # Uncertainty-gated fairness: boost waiting time only when slack > 0, dampened by uncertainty
    fairness_base = np.where(slack > 0, ready_wait_time / (1.0 + uncertainty + eps), 0.0)

    # Upward-rank weighted latency pressure (critical path latency sensitivity)
    latency_pressure = upward_rank * exec_comm_sum / (np.clip(slack, eps, 1e6) + eps)

    # Assemble raw components before normalization
    raw_terms = {
        'urgency': urgency,
        'latency_resilience': latency_resilience,
        'energy': energy_term,
        'fairness': fairness_base,
        'latency_pressure': latency_pressure
    }

    # IQR-based robust normalization per term: x -> (x - Q1) / (Q3 - Q1 + eps)
    def iqr_normalize(x):
        x = np.clip(x, -1e6, 1e6)
        if x.size == 1:
            return np.array([0.0])
        q1 = np.percentile(x, 25)
        q3 = np.percentile(x, 75)
        iqr = q3 - q1
        if iqr < eps:
            return np.zeros_like(x)
        return (x - q1) / (iqr + eps)

    norm_terms = {k: iqr_normalize(v) for k, v in raw_terms.items()}

    # Auto-weighting: weight ∝ std(norm_term) to emphasize higher-variance signals
    weights = {k: np.std(v) + eps for k, v in norm_terms.items()}
    total_weight = sum(weights.values())
    weights = {k: w / total_weight for k, w in weights.items()}

    # Final score: minimize → smaller = better. Urgency & latency_pressure are *positive urgency*
    # Energy is negated when beneficial (SEER), so lower energy_term improves score → keep sign as-is
    score = (
        weights['urgency'] * norm_terms['urgency'] +
        weights['latency_resilience'] * norm_terms['latency_resilience'] +
        weights['energy'] * norm_terms['energy'] +
        weights['fairness'] * norm_terms['fairness'] +
        weights['latency_pressure'] * norm_terms['latency_pressure']
    )

    # Hard violation override: any task with slack < -eps gets top priority (lowest score)
    violation_mask = slack < -eps
    if np.any(violation_mask):
        base_ref = np.min(score[~violation_mask]) if np.any(~violation_mask) else np.min(score)
        score = np.where(violation_mask, base_ref - 1e9, score)

    # Final sanitization
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)

    # Ensure shape (N,) — no squeezing
    return score.reshape(-1)
