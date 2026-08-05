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
    v2: Hybrid deadline-hardened, energy-aware, numerically robust priority scorer.
    Combines Parent 2's quantile normalization, urgency-weighted energy fairness,
    exponential aging, and adaptive latency windows — with Parent 1's slack-gated
    energy activation, hard-DDL offset for overdue tasks, and uncertainty penalty
    scaled by normalized urgency. Introduces unified risk-aware latency footprint
    and finite-range clamping for numerical stability.
    """
    eps = 1e-08
    # Safe conversion and nan/inf cleanup
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=0.0, posinf=1e6, neginf=1e6)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=0.0, posinf=1e6, neginf=1e6)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=0.0, posinf=1e6, neginf=1e6)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=0.0, posinf=1e6, neginf=1e6)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=0.0, posinf=1e6, neginf=1e6)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=1e6, neginf=1e6)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=0.0, posinf=1e6, neginf=1e6)

    def robust_quantile_normalize(x):
        x = np.clip(x, -1e6, 1e6)
        if x.size == 0:
            return np.zeros_like(x)
        if x.size == 1:
            return np.array([0.0])
        q1, q3 = np.quantile(x, [0.25, 0.75], method='midpoint')
        iqr = q3 - q1
        if iqr < eps:
            return np.zeros_like(x)
        # Clip to IQR bounds before normalization to suppress outliers
        x_clipped = np.clip(x, q1, q3)
        return (x_clipped - q1) / (iqr + eps)

    # === Urgency & Hard-DDL Handling ===
    # Raw urgency: absolute lateness (0 when slack >= 0)
    urgency_raw = np.where(slack < 0, -slack, 0.0)
    norm_urgency = robust_quantile_normalize(urgency_raw)
    # Hard offset for overdue tasks (prevents under-prioritization due to normalization)
    hard_ddl_offset = np.where(slack < 0, 120.0, 0.0)

    # === Critical Path & Latency Footprint ===
    # Total latency footprint (execution + communication), robustly bounded
    latency_footprint = np.maximum(min_exec_time + min_comm_time, eps)
    critical_density = upward_rank * latency_footprint
    norm_critical_density = robust_quantile_normalize(critical_density)

    # === Energy Efficiency with Slack-Gated Activation ===
    # Energy per unit work — only activated when slack is safe (>= median_slack)
    energy_per_work = min_incremental_energy / (remaining_work + eps)
    median_slack = np.median(slack) + eps
    energy_gate = (slack >= median_slack).astype(float)
    norm_energy_eff = robust_quantile_normalize(energy_per_work) * energy_gate
    # Urgency-weighted boost: when overdue, prioritize energy-efficient tasks more strongly
    urgency_factor = np.where(slack >= 0, 1.0, 1.0 + 0.4 * (urgency_raw + eps))
    energy_term = norm_energy_eff * urgency_factor

    # === Uncertainty Penalty: Risk-amplified by urgency ===
    # Active only when slack < 0 AND uncertainty > median_uncertainty
    median_uncert = np.median(uncertainty) + eps
    risk_active = (slack < 0) & (uncertainty > median_uncert)
    uncertainty_penalty = np.where(risk_active, uncertainty * norm_urgency, 0.0)
    norm_uncert_penalty = robust_quantile_normalize(uncertainty_penalty)

    # === Aging Fairness: Exponential decay gated by deadline pressure ===
    # Only applied under severe deadline stress (slack < -0.1s)
    aging_mask = (slack < -0.1).astype(float)
    tau = np.maximum(eps, np.median(ready_wait_time[ready_wait_time > 0]) if np.any(ready_wait_time > 0) else 1.0)
    aging_raw = np.where(
        aging_mask == 1,
        1.0 - np.exp(-np.clip(ready_wait_time, 0.0, 1000.0) / (tau + eps)),
        0.0
    )
    aging_norm = robust_quantile_normalize(aging_raw)
    aging_term = -aging_norm * urgency_raw * aging_mask  # Boost priority of long-waiting overdue tasks

    # === Latency Risk: Adaptive arctan window based on mean base latency ===
    base_latency = latency_footprint
    mean_base_latency = np.mean(base_latency) + eps
    adaptive_window = np.maximum(0.5, 0.1 * mean_base_latency)
    imminent_window = (slack >= 0) & (slack < adaptive_window)
    relative_volatility = base_latency * (uncertainty + eps) / mean_base_latency
    risk_arctan = np.where(imminent_window, 0.5 + 1.0 / np.pi * np.arctan(relative_volatility), 0.0)
    latency_risk_term = robust_quantile_normalize(risk_arctan)

    # === Final weighted score (smaller = better) ===
    # Weights tuned for deadline hardness first, then energy & fairness
    w_urgency = 6.0          # Dominant for overdue tasks
    w_critical = 2.0         # Critical path importance
    w_energy = 2.2           # Energy efficiency, amplified under pressure
    w_uncert = 1.3           # Uncertainty penalty under risk
    w_aging = 1.1            # Starvation relief under deadline stress
    w_risk = 1.8             # Latency risk near soft deadline
    w_offset = 1.0           # Hard offset multiplier (applied as-is)

    score = (
        w_urgency * norm_urgency +
        w_critical * norm_critical_density +
        w_energy * energy_term +
        w_uncert * norm_uncert_penalty +
        w_aging * aging_term +
        w_risk * latency_risk_term +
        w_offset * hard_ddl_offset
    )

    # Final sanitization: ensure finite, deterministic, shape-correct output
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)
    # Ensure shape (N,) even for N=1
    if score.ndim == 0:
        score = np.array([score])
    elif score.ndim > 1:
        score = score.reshape(-1)

    return score
