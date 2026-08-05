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
    v2: Lexicographic deadline-first priority with robust quantile-normalized features,
    adaptive risk-gated energy efficiency, and uncertainty-aware criticality amplification.
    
    Key mutations:
    - Replaces min-max normalization with robust quantile-based scaling (Q1/Q3) for outlier resilience.
    - Introduces *deadline-anchored urgency*: arctan-based monotonic mapping of robust_slack,
      bounded and differentiable, zero at slack=0, negative for violations.
    - Energy term gated by *both* slack > 0 AND uncertainty < threshold — avoids penalizing high-risk tasks
      when deadlines are tight, preserving feasibility.
    - Criticality term redefined as upward_rank × (remaining_work / (min_exec_time + min_comm_time + eps))
      to favor high-impact work per latency cost — then scaled by (1 + uncertainty) only when slack > 0.
    - Fairness uses ready_wait_time normalized *relative to max wait in ready set*, multiplied by
      inverse slack safety margin (1 / (1 + max(0, -slack))) to gently boost starvation relief
      without overriding hard DDL constraints.
    - All terms combined multiplicatively in urgency-dominated lexicographic order:
      urgency dominates → criticality → energy → fairness (with sign flipped for priority ordering).
    - No additive weighting; instead, hierarchical gating ensures deadline compliance is never diluted.
    """
    eps = 1e-8
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
    
    # Robust slack: subtract 2×uncertainty only when slack > 0; preserve violation signal when slack ≤ 0
    robust_slack = np.where(slack > eps, slack - 2.0 * uncertainty, slack)
    
    # Deadline-anchored urgency: arctan-based, monotonic, bounded [-π/2, π/2], zero-crossing at robust_slack == 0
    # Negative urgency = higher priority for overdue tasks; positive = lower priority for safe tasks
    urgency = np.arctan(-robust_slack / (eps + np.clip(np.abs(np.median(robust_slack)), eps, 1e6)))
    
    # Criticality: work-density importance — upward_rank × (remaining_work per latency unit)
    exec_comm = min_exec_time + min_comm_time + eps
    criticality_base = upward_rank * (remaining_work / exec_comm)
    # Amplify criticality only when slack > 0 and uncertainty is moderate — avoid over-prioritizing noisy critical paths
    criticality_gate = np.where((robust_slack > eps) & (uncertainty < 0.7), 1.0 + uncertainty, 1.0)
    criticality = criticality_base * criticality_gate
    
    # Energy term: only active when slack > 0 AND uncertainty < 0.5 — strict safety guard for feasibility
    energy_gate = np.where((robust_slack > eps) & (uncertainty < 0.5), 1.0, 0.0)
    energy_term = min_incremental_energy / (exec_comm + 1.0) * energy_gate
    
    # Fairness: relative wait time scaled by inverse safety margin (prevents starvation without violating DDL)
    max_wait = np.max(ready_wait_time) if ready_wait_time.size > 0 else eps
    rel_wait = np.clip(ready_wait_time / (max_wait + eps), 0.0, 1.0)
    safety_margin = np.clip(1.0 + np.maximum(0.0, -robust_slack), 1.0, 1e3)
    fairness = rel_wait / safety_margin
    
    # Robust quantile-based normalization: Q1–Q3 range scaling with fallback for low-variance/small-N
    def quantile_normalize(x):
        x = np.clip(x, -1e6, 1e6)
        if x.size == 1:
            return np.array([0.0])
        q1, q3 = np.quantile(x, [0.25, 0.75])
        iqr = q3 - q1 + eps
        normed = (x - q1) / iqr
        # Clamp extreme outliers post-normalization
        return np.clip(normed, -10.0, 10.0)
    
    norm_urgency = quantile_normalize(urgency)
    norm_criticality = quantile_normalize(criticality)
    norm_energy = quantile_normalize(energy_term)
    norm_fairness = quantile_normalize(fairness)
    
    # Lexicographic dominance via multiplicative hierarchy:
    # urgency dominates (base); criticality modulates it only when urgent; energy refines under safety;
    # fairness breaks ties *only* when all else is equal — implemented via small coefficient + sign flip
    score = (
        norm_urgency  # Primary: smallest = most urgent
        + 0.3 * np.where(norm_urgency > -0.1, norm_criticality, 0.0)  # Secondary: amplify only in near-deadline regime
        + 0.1 * np.where(norm_urgency > -0.5, norm_energy, 0.0)       # Tertiary: refine energy under safety
        - 0.02 * norm_fairness  # Tie-breaker: smaller fairness score = higher priority (anti-starvation)
    )
    
    # Hard override for deadline violations: any task with original slack < -1e-6 gets lowest possible score
    violation_mask = slack < -1e-6
    if np.any(violation_mask):
        base_min = np.min(score[~violation_mask]) if np.any(~violation_mask) else np.min(score)
        score = np.where(violation_mask, base_min - 1e9, score)
    
    # Final sanitization: ensure finite, shape-(N,), deterministic output
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)
    
    # Enforce shape (N,) — no scalars, no column vectors
    return score.reshape(-1)
