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
    Priority rule v2: Lexicographic deadline-feasibility first, then energy efficiency,
    with robust signal preservation and starvation prevention.
    
    Key innovations:
    - Urgency term uses softplus(-slack) for smooth, monotonic, wide-dynamic-range penalty;
      normalized via MAD with z-score fallback for degenerate cases.
    - SEER (Energy per Effective Runtime) is gated *multiplicatively* by urgency mask
      (1 if slack >= 0, else 0), enforcing strict deadline priority before energy optimization.
    - Criticality-pressure combines upward_rank and remaining_work *before* risk scaling,
      normalized jointly to avoid rank inflation; uncertainty modulates via tanh, not raw scale.
    - Fairness uses linear aging ramp + bounded sigmoid wait-ratio, scaled by |slack|+1 to
      prevent over-prioritization under extreme tightness.
    - All normalizations are adaptive, finite, and safe for N=1; final score is clipped and sanitized.
    """
    eps = 1e-8
    # Safely cast and sanitize all inputs
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=1e6, neginf=eps)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=0.0, posinf=100.0, neginf=0.0)

    def robust_normalize(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        if mad < eps:
            var = np.var(x)
            if var < eps:
                return np.zeros_like(x, dtype=float)
            return (x - med) / (np.sqrt(var) + eps)
        return (x - med) / (mad + eps)

    # Urgency: softplus(-slack) → smooth, monotonic, unbounded-from-below penalty for lateness
    urgency_raw = np.log1p(np.exp(-slack))  # ≈ max(0, -slack) + smooth tail
    norm_urgency = robust_normalize(urgency_raw)
    # Scale to [0.1, 15.0] for strong lexicographic dominance
    urgency_term = np.clip(0.1 + 14.9 * (norm_urgency - np.min(norm_urgency) + eps) / 
                          (np.max(norm_urgency) - np.min(norm_urgency) + eps), 0.1, 15.0)

    # SEER: Energy per effective runtime — only active when slack >= 0 (deadline-safe)
    seer_base = (min_incremental_energy + eps) / (min_exec_time + min_comm_time + eps)
    slack_gate = (slack >= 0).astype(float)  # binary gate: 1 if safe, 0 if at-risk
    seer_gated = seer_base * slack_gate
    norm_seer = robust_normalize(seer_gated)
    # Invert and compress: higher SEER → lower priority (worse energy efficiency)
    seer_term = np.clip(1.0 - 0.7 * norm_seer, 0.05, 2.0)

    # Criticality-pressure: joint importance of rank + work, risk-modulated
    critical_base = upward_rank * remaining_work
    risk_factor = 1.0 + np.tanh(uncertainty)  # bounded [1.0, 2.0]
    critical_risk = critical_base * risk_factor
    norm_critical = robust_normalize(critical_risk)
    # Higher criticality → higher priority → lower score → negative weight
    critical_term = -0.85 * norm_critical

    # Fairness (aging): prevents starvation; scales with wait but damped by deadline pressure
    wait_ratio = (ready_wait_time + eps) / (np.abs(slack) + 1.0 + eps)
    # Bounded sigmoid + linear ramp for monotonic, interpretable aging
    sigmoid_part = 1.0 / (1.0 + np.exp(-wait_ratio + 1.5))
    linear_part = np.clip(ready_wait_time * 0.03, 0.0, 0.7)
    combined_aging = 0.6 * sigmoid_part + 0.4 * linear_part
    norm_aging = robust_normalize(combined_aging)
    aging_term = 0.15 * norm_aging

    # Final score: multiplicative urgency × SEER ensures deadline feasibility dominates energy,
    # additive terms refine within feasibility region
    score = urgency_term * seer_term + critical_term + aging_term

    # Ensure finite, deterministic output
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)
    return score.astype(float)
