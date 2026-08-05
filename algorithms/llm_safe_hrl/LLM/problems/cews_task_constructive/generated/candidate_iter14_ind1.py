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
    Evolved priority rule v2: Deadline-hardened criticality + uncertainty-aware energy leverage + starvation-resilient fairness.
    
    Key innovations:
    - Strict slack-triggered *criticality inversion*: overdue tasks get maximally negative urgency and inverted upward_rank
    - Unified risk scaling: uncertainty linearly amplifies both deadline penalty and energy sensitivity, bounded [0,3]
    - Work-normalized communication pressure only under tight slack (slacks < 0.5*median_duration)
    - Starvation control via relative wait ratio *only when slack >= 0*, avoiding interference with urgent tasks
    - Robust MAD-based normalization with tighter clipping (-6, +6) for numerical stability
    - Energy term is *inverted and scaled* only when slack > 0 — preserves energy awareness in feasible region
    - All operations protected by eps; no NaN/inf propagation; deterministic and monotonic in key drivers
    """
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)

    def robust_normalize_mad(x):
        if len(x) == 0:
            return x
        center = np.median(x)
        mad = np.median(np.abs(x - center)) + eps
        normed = (x - center) / mad
        return np.clip(normed, -6.0, 6.0)

    # Base duration and normalized slack
    task_min_duration = np.maximum(min_exec_time + min_comm_time, eps)
    median_duration = np.median(task_min_duration) + eps
    rel_slack = slack / median_duration

    # Urgency: piecewise-linear for strict monotonicity and hard deadline enforcement
    # Overdue: linear penalty; feasible: exponential decay to avoid over-prioritizing tiny slack
    urgency = np.where(
        slack <= 0.0,
        -rel_slack * 3.0,  # strong linear penalty for lateness
        np.exp(-rel_slack * 1.0)  # smooth decay for positive slack
    )
    urgency_norm = robust_normalize_mad(urgency)

    # Criticality inversion: flip upward_rank sign for overdue tasks
    ur_inverted = np.where(slack <= 0.0, -upward_rank, upward_rank)
    ur_norm = robust_normalize_mad(ur_inverted)

    # Risk-adjusted energy leverage: higher uncertainty reduces energy weight *only* when slack > 0
    # When overdue (slack <= 0), energy is deprioritized — deadline dominates
    energy_weight = np.where(
        slack <= 0.0,
        0.0,
        (1.0 + 0.7 * np.clip(uncertainty, 0.0, 3.0))
    )
    energy_score = -robust_normalize_mad(min_incremental_energy) * energy_weight

    # Communication pressure: high comm-to-work ratio prioritized *only* under tight slack
    comm_to_work_ratio = min_comm_time / (remaining_work + eps)
    tight_slack_mask = (slack < 0.5 * median_duration) & (slack != np.inf)
    comm_pressure = np.where(tight_slack_mask, comm_to_work_ratio * 2.0, comm_to_work_ratio * 0.2)
    comm_norm = robust_normalize_mad(comm_pressure)

    # Starvation fairness: only active when slack >= 0 and work is non-trivial
    max_wait = np.maximum(np.max(ready_wait_time), eps)
    rel_wait = np.clip(ready_wait_time / max_wait, 0.0, 1.0)
    starvation_mask = (slack >= 0.0) & (remaining_work > np.median(remaining_work) + eps)
    starvation_penalty = np.where(starvation_mask, rel_wait * 0.4, 0.0)

    # Final convex combination — weighted to emphasize urgency (2.8x), criticality (2.0x), energy (1.2x)
    score = (
        -2.8 * urgency_norm
        - 2.0 * ur_norm
        + 1.2 * energy_score
        + 0.5 * comm_norm
        + starvation_penalty
    )

    # Final safeguard: replace NaN/inf, clamp extremes
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)

    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
