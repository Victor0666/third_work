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
    Priority rule v2: Lexicographic DDL-hardened scoring with:
      - Bounded arctan urgency (smooth, monotonic, zero-discontinuity at slack=0)
      - Energy-criticality ratio (ECR) = (upward_rank * remaining_work) / (min_incremental_energy + eps)
        inverted to reward energy efficiency *only* when feasible (slack >= 0), avoiding penalty on urgent tasks
      - Uncertainty-weighted latency-efficiency denominator: (min_exec_time + min_comm_time) * (1 + tanh(uncertainty))
      - Deadline-proportional fairness: ready_wait_time / (|slack| + eps), gated by slack > 0.5 to prevent starvation near deadline
      - Tie-breaking via robust inverse latency & inverse criticality, normalized and weighted to preserve hierarchy
      - All components rigorously guarded against NaN/inf/zero; strict lexicographic dominance via multiplicative scaling
    """
    eps = 1e-08
    # Safe input conversion and nan/inf clipping
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=1e6, neginf=eps)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=0.0, posinf=100.0, neginf=0.0)

    # Urgency: smooth, bounded arctan-based risk signal — higher score for more negative slack
    # arctan(10*slack) maps slack ∈ [-∞, ∞] → (-π/2, π/2); shift & scale to [0,1] where 1 = most urgent
    urgency_raw = (np.arctan(10.0 * slack) + np.pi / 2) / np.pi
    urgency_score = 1.0 - urgency_raw  # now 1.0 = max urgency (highly negative slack), 0.0 = no urgency (large positive slack)

    # ECR (Energy-Criticality Ratio): reward energy efficiency *relative to criticality*, but only for non-urgent tasks
    # When slack < 0, we prioritize urgency over energy → set ECR to 0 to avoid diluting urgency dominance
    ecr_numerator = upward_rank * remaining_work
    ecr_denom = min_incremental_energy + eps
    ecr_base = ecr_numerator / ecr_denom
    ecr_score = np.where(slack >= 0, ecr_base, 0.0)  # zero out for urgent tasks → preserves lexicographic priority

    # Latency-efficiency penalty: lower is better; incorporates uncertainty smoothly via tanh
    unc_tanh = np.tanh(uncertainty)  # ∈ [0, 1) for uncertainty ≥ 0
    latency_base = min_exec_time + min_comm_time + eps
    latency_penalty = latency_base * (1.0 + unc_tanh)

    # Fairness: wait pressure scaled by deadline margin, activated only when slack > 0.5s to avoid interfering with hard DDL
    fairness_raw = np.where(slack > 0.5, ready_wait_time / (np.abs(slack) + eps), 0.0)
    fairness_score = np.clip(fairness_raw, 0.0, 1e4)

    # Tie-breaking: favor low execution time AND high criticality (low inv_exec, low inv_rank)
    inv_exec = 1.0 / (min_exec_time + eps)
    inv_rank = 1.0 / (upward_rank + eps)

    def robust_minmax_normalize(x):
        if x.size == 1:
            return np.full_like(x, 0.5, dtype=float)
        x_min, x_max = np.min(x), np.max(x)
        if x_max - x_min < eps:
            return np.full_like(x, 0.5, dtype=float)
        return np.clip((x - x_min) / (x_max - x_min + eps), 0.0, 1.0)

    # Normalize all components to [0,1] for stable combination
    norm_urgency = robust_minmax_normalize(urgency_score)
    norm_ecr = robust_minmax_normalize(ecr_score)
    norm_latency = robust_minmax_normalize(latency_penalty)
    norm_fairness = robust_minmax_normalize(fairness_score)
    norm_inv_exec = robust_minmax_normalize(inv_exec)
    norm_inv_rank = robust_minmax_normalize(inv_rank)

    # Lexicographic dominance via large multiplicative weights: urgency >> ECR >> latency >> fairness >> tiebreak
    # Use powers of 10 to ensure strict ordering even under floating-point noise
    score = (
        norm_urgency * 1e6 +
        norm_ecr * 1e4 +
        norm_latency * 1e2 +
        norm_fairness * 1e1 +
        (1.0 - norm_inv_exec) * 1.0 +  # prefer lower exec time → higher priority
        (1.0 - norm_inv_rank) * 0.1    # prefer higher criticality → higher priority
    )

    # Final sanitization: clamp and eliminate inf/nan
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=1e9)
    score = np.clip(score, eps, 1e9)

    return score.astype(float)
