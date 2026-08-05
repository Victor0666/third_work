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
    v2: Mutated priority rule emphasizing risk-aware deadline urgency, 
    physics-aligned energy-latency tradeoff, and MAD-based robust normalization.
    
    Key mutations vs v1:
    - Replaces z-score with MAD (median absolute deviation) scaling for outlier resilience
    - Uses arctan-based urgency mapping instead of clipped tanh: smoother, bounded [-π/2, π/2]
    - Introduces "criticality-pressure" term: upward_rank × (1 + uncertainty) / (min_exec_time + eps)
      to amplify importance of high-rank tasks under uncertainty
    - Replaces multiplicative fairness gating with additive aging bonus *only* when slack > 0,
      scaled by normalized wait time and capped at 5% of base score
    - Energy term now uses SEER-like ratio: (min_exec_time + min_comm_time) / (min_incremental_energy + eps),
      inverted to favor low-energy-per-latency tasks
    - Hard violation override now assigns -inf (not finite negative) for strict lexicographic dominance
    - All terms composed additively after robust normalization (not multiplicatively) to avoid
      compounding bias and improve gradient stability during learning
    - Final score clamped to finite bounds and sanitized against inf/nan
    """
    eps = 1e-8
    
    def sanitize(x):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(x, nan=eps, posinf=1e6, neginf=1e-6)
    
    min_exec_time = sanitize(min_exec_time)
    min_comm_time = sanitize(min_comm_time)
    min_incremental_energy = sanitize(min_incremental_energy)
    slack = sanitize(slack)
    upward_rank = sanitize(upward_rank)
    remaining_work = sanitize(remaining_work)
    ready_wait_time = sanitize(ready_wait_time)
    uncertainty = sanitize(uncertainty)
    
    # Robust slack: shift by 2×uncertainty to absorb estimation noise
    robust_slack = slack - 2.0 * uncertainty
    robust_slack = np.clip(robust_slack, -1e5, 1e5)
    
    # Arctan urgency: smooth, bounded, monotonic, avoids clipping artifacts
    # Maps robust_slack ∈ (-∞, ∞) → urgency ∈ (-π/2, π/2), then shift+scale to [0, 1]
    urgency_raw = np.arctan(robust_slack / (np.abs(robust_slack) + eps))
    urgency_norm = (urgency_raw + np.pi/2) / np.pi  # [0, 1] where 0 = most urgent
    
    # Criticality-pressure: high upward_rank + high uncertainty + short exec → boost priority
    # Avoids division-by-zero; scales inversely with execution time (faster tasks on CP get push)
    cp_pressure = upward_rank * (1.0 + uncertainty) / (min_exec_time + eps)
    
    # Latency-energy efficiency ratio (SEER analog): higher = more energy-efficient per latency unit
    # Inverted so lower score = better (since we minimize score)
    seer_ratio = (min_exec_time + min_comm_time + eps) / (min_incremental_energy + eps)
    
    # Fairness: additive aging bonus only when slack > 0, capped at 5% of base urgency contribution
    aging_bonus = np.where(slack > 0.0, 
                          np.clip(ready_wait_time / (np.abs(slack) + eps), 0.0, 10.0), 
                          0.0)
    aging_bonus = np.clip(aging_bonus * 0.05 * (1.0 - urgency_norm), 0.0, 0.05)
    
    # Robust MAD-based normalization (more stable than std for small/N=1 sets)
    def normalize_mad(x):
        x = np.clip(x, -1e6, 1e6)
        if x.size == 1:
            return np.array([0.0])
        median = np.median(x)
        mad = np.median(np.abs(x - median))
        if mad < eps:
            return np.zeros_like(x)
        return (x - median) / (mad + eps)
    
    norm_urgency = normalize_mad(urgency_norm)
    norm_cp = normalize_mad(cp_pressure)
    norm_seer = normalize_mad(seer_ratio)
    
    # Additive composition: preserves interpretability & avoids multiplicative explosion
    # Urgency dominates (negative weight), CP adds pressure, SEER adds efficiency incentive
    score = (
        -2.0 * norm_urgency   # Strongest weight: deadline urgency is primary driver
        + 0.8 * norm_cp       # Moderate weight: critical path pressure second
        + 0.5 * norm_seer     # Light weight: energy-latency efficiency third
        + aging_bonus         # Small fairness correction, only in feasible region
    )
    
    # Hard violation override: assign -inf for any task with slack < 0 → guaranteed highest priority
    violation_mask = (slack < 0.0)
    score = np.where(violation_mask, -np.inf, score)
    
    # Final sanitization: ensure finite output, shape (N,)
    score = np.nan_to_num(score, nan=1e6, posinf=1e6, neginf=-1e6)
    score = np.clip(score, -1e6, 1e6)
    
    return score
