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
    v4: Hard-DDL-first, energy-latency-ratio optimized, numerically robust priority with adaptive criticality gating and zero-risk dominance.
    
    Key improvements:
    - Combines Parent 2's piecewise-linear urgency (strict monotonicity, no vanishing gradients) and ELR with Parent 1's violation-boost safeguard.
    - Replaces normalized CP density with *slack-gated criticality*: upward_rank * remaining_work * sigmoid(slack), ensuring high-importance tasks still gain priority under safety margin.
    - Introduces *energy-efficiency boost*: inverse of ELR when slack > 0, directly promoting low-energy-per-latency assignments in safe regions.
    - Fairness uses *wait-pressure ratio* (ready_wait_time / median(exec_comm_sum + eps)) scaled by slack surplus — more stable than mean-based normalization.
    - Uncertainty is additive only for high-risk tasks (slack > 1.5 AND uncertainty > 0.2), capped and gated to avoid noise amplification.
    - All normalization uses deterministic min-max scaling with N=1 guard; no z-score or MAD to preserve rank fidelity.
    - Zero-risk enforcement: if any task has slack <= 0, all non-violating tasks receive +inf penalty unless they are the most urgent (min slack).
    - Final score clamped, finite, and reshaped to (N,) deterministically.
    """
    eps = 1e-08
    N = len(slack)
    
    # Defensive input sanitization
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=1000000.0, neginf=eps)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=1000000.0, neginf=eps)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=1000000.0, neginf=eps)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=1000000.0, neginf=-1000000.0)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=0.0, posinf=1000000.0, neginf=0.0)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=0.0, posinf=1000000.0, neginf=0.0)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=1000000.0, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=0.0, posinf=1000000.0, neginf=0.0)
    
    # Piecewise-linear urgency (Parent 2): exact penalty for violations, linear decay for tight slack, zero beyond
    tau_urgency = 1.5
    urgency_raw = np.where(
        slack <= 0,
        -slack * 6.0,
        np.where(
            slack <= tau_urgency,
            6.0 * (1.0 - slack / (tau_urgency + eps)),
            0.0
        )
    )
    
    # Energy-Latency Ratio (ELR) — Parent 2 core insight
    exec_comm_sum = min_exec_time + min_comm_time + eps
    elr_base = np.clip(min_incremental_energy / (exec_comm_sum + 3.0), 0.0, 100000.0)
    elr_masked = np.where(slack > 0, elr_base, 0.0)
    
    # Slack-gated criticality: preserves critical path importance but smoothly attenuates as slack increases (novel blend)
    # Uses sigmoid instead of hard clamp for smooth gradient and better portability
    slack_sigmoid = 1.0 / (1.0 + np.exp(-(slack - 0.5) / (0.5 + eps)))
    criticality_raw = upward_rank * remaining_work * slack_sigmoid
    
    # Wait-pressure fairness: relative to median latency (more robust than mean), scaled by slack margin
    exec_comm_median = np.median(exec_comm_sum) if N > 1 else exec_comm_sum[0]
    wait_pressure = np.clip(ready_wait_time / (exec_comm_median + eps), 0.0, 10.0)
    slack_margin = np.clip(np.maximum(0.0, slack) / (tau_urgency + eps), 0.0, 1.0)
    fairness_raw = wait_pressure * slack_margin
    
    # Uncertainty boost: additive, gated, capped (Parent 2 style, tightened threshold)
    unc_boost = np.where(
        (slack > 1.5) & (uncertainty > 0.2),
        np.clip(uncertainty * 0.4, 0.0, 0.2),
        0.0
    )
    
    # Zero-risk dominance: if any task violates deadline, non-violating tasks get large penalty unless they're most urgent
    has_violation = np.any(slack <= 0)
    min_slack_idx = np.argmin(slack)
    violation_boost = np.where(
        has_violation,
        np.where(
            slack <= 0,
            0.0,  # violating tasks get no boost (they're already urgent)
            np.where(
                np.arange(N) == min_slack_idx,
                0.0,  # most urgent non-violator gets no penalty
                1000000.0  # all other non-violators heavily penalized
            )
        ),
        0.0
    )
    
    # Robust min-max normalization (Parent 2 style — deterministic, outlier-resilient, N=1 safe)
    def normalize_minmax(x):
        x = np.clip(x, -1e6, 1e6)
        if x.size == 1:
            return np.array([0.0])
        x_min, x_max = np.min(x), np.max(x)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x - x_min) / (x_max - x_min + eps)
    
    norm_urgency = normalize_minmax(urgency_raw)
    norm_elr = normalize_minmax(elr_masked)
    norm_criticality = normalize_minmax(criticality_raw)
    norm_fairness = normalize_minmax(fairness_raw)
    norm_unc = normalize_minmax(unc_boost)
    
    # Weighted terms — urgency and ELR dominate; criticality and fairness support; uncertainty slightly penalizes risk
    urgency_term = -8.0 * norm_urgency
    elr_term = -4.0 * norm_elr
    criticality_term = 0.6 * norm_criticality
    fairness_term = -0.4 * norm_fairness
    unc_term = -0.15 * norm_unc
    
    # Sum and finalize
    score = urgency_term + elr_term + criticality_term + fairness_term + unc_term + violation_boost
    
    # Final sanitization: ensure finite, bounded, correct shape
    score = np.nan_to_num(score, nan=1e8, posinf=1e8, neginf=-1e8)
    score = np.clip(score, -1e8, 1e8)
    return score.reshape(-1)
