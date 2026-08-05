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
    Evolved priority rule: combines Parent 2's robustness and deadline smoothness
    with Parent 1's explicit waiting fairness and uncertainty-aware latency damping.
    Key innovations:
      - Uses arctan-based deadline risk (Parent 2) but adds hard penalty boost when any task is late
      - Critical-energy efficiency gated by slack > 0 (Parent 2), normalized via MAD for stability
      - Waiting boost uses sqrt-based fairness scaled by slack margin AND uncertainty-dampened (hybrid)
      - Uncertainty inflates latency only for slack > 0, avoiding over-penalization of risky-but-urgent tasks
      - All normalizations use safe_mad_normalize with clipping; all divisions guarded by eps
      - Final score strictly prioritizes DDL compliance first: negative slack triggers dominant term
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

    def safe_mad_normalize(x):
        """Robust MAD normalization: handles N=1, constants, outliers; clips to [-4,4]"""
        if x.size == 1:
            return np.zeros_like(x)
        med = np.median(x)
        abs_dev = np.abs(x - med)
        mad = np.median(abs_dev) + eps
        normed = (x - med) / mad
        return np.clip(normed, -4.0, 4.0)

    # Deadline risk: bounded arctan for smooth penalty, boosted if ANY task is late
    deadline_risk_raw = np.where(slack < 0, np.arctan(-slack), 0.0)
    deadline_score = safe_mad_normalize(deadline_risk_raw)
    has_late_task = np.any(slack < 0)
    deadline_factor = 5.0 if has_late_task else 1.0  # Stronger penalty when deadlines violated

    # Critical-energy efficiency: only active when slack > 0 (DDL-safe regime)
    critical_work_gated = np.where(slack > 0, upward_rank * remaining_work, 0.0)
    critical_work_norm = safe_mad_normalize(critical_work_gated)
    energy_denom = min_incremental_energy + eps
    critical_energy_ratio = np.where(slack > 0, (upward_rank * remaining_work + eps) / energy_denom, 0.0)
    # Sigmoid bounded to [0,1] for stable energy-efficiency signal
    critical_energy_sigmoid = 1.0 / (1.0 + np.exp(-np.clip(critical_energy_ratio, -5.0, 5.0) * 0.3))
    critical_energy_norm = safe_mad_normalize(critical_energy_sigmoid)

    # Waiting fairness: sqrt-based boost, scaled by slack margin AND dampened by uncertainty
    wait_boost_base = np.sqrt(np.clip(ready_wait_time, 0.0, None))
    slack_margin = np.clip(slack, 0.0, None) + eps
    # Uncertainty reduces urgency of waiting boost (high uncertainty → less trust in wait time)
    wait_boost = np.clip(wait_boost_base / (slack_margin + wait_boost_base + eps), 0.0, 0.2)
    wait_boost_damped = wait_boost / (1.0 + np.clip(uncertainty, 0.0, 2.0) * 0.5 + eps)

    # Latency inflation: only applied in DDL-safe region, uncertainty-modulated
    total_latency = min_exec_time + min_comm_time + eps
    uncertainty_inflated_latency = np.where(
        slack > 0,
        total_latency * (1.0 + np.clip(uncertainty, 0.0, 2.0) * 0.2),
        total_latency
    )
    inflated_lat_norm = safe_mad_normalize(uncertainty_inflated_latency)

    # Remaining work contributes weakly as proxy for load balancing
    remaining_work_norm = safe_mad_normalize(remaining_work) * 0.03

    # Final score: smaller = higher priority; deadline dominates when violated
    score = (
        +deadline_factor * 4.0 * deadline_score
        - 2.0 * critical_work_norm
        + 0.8 * critical_energy_norm
        + 0.18 * wait_boost_damped
        + 0.22 * inflated_lat_norm
        + remaining_work_norm
    )

    # Ensure finite output: replace NaN/inf with large finite values (higher priority penalty)
    return np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
