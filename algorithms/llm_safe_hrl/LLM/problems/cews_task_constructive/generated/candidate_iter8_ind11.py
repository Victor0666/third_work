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
    v2: Deadline-hardened, energy-aware, numerically bulletproof priority scorer.
    Combines Parent 2's strict urgency linearity and median-based fairness with Parent 1's
    robust arctan risk scaling (adapted to imminent window only) and clipped sqrt-wait fairness.
    Key novelties:
    - Urgency: linear ramp for slack < 0, zero otherwise (strict deadline dominance)
    - Energy density: normalized by *median* (not min-max) to resist outliers; gated by slack >= 0
    - Latency-risk: arctan-scaled volatility only for 0 <= slack < 1.0 (imminent window), avoiding saturation
    - Fairness: clipped sqrt(wait) term activated *only* when slack < -0.1s (aging under deadline pressure)
    - Proximity penalty: linear 1-slack in [0,1), normalized robustly
    - All divisions guarded, NaN/inf handled, shape-(N,) guaranteed, deterministic.
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

    def robust_minmax_normalize(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 0:
            return np.zeros_like(x)
        if x.size == 1:
            return np.array([0.0])
        x_min, x_max = np.min(x), np.max(x)
        range_val = x_max - x_min
        if range_val < eps:
            return np.zeros_like(x)
        return (x - x_min) / (range_val + eps)

    # === 1. Strict deadline dominance: linear urgency for overdue tasks only ===
    urgency_raw = np.where(slack < 0, -slack, 0.0)
    deadline_score = robust_minmax_normalize(urgency_raw)

    # === 2. Energy-density fairness: critical work per marginal joule, median-normalized ===
    # Use median normalization (Parent 2) for outlier resilience, not subset min-max (Parent 1)
    energy_density = upward_rank * (remaining_work + eps) / (min_incremental_energy + eps)
    # Compute median on full array, then normalize — avoids skew from subset filtering
    med_energy = np.median(energy_density) if energy_density.size > 0 else 1.0
    energy_density_centered = np.abs(energy_density - med_energy) + eps
    # Normalize relative to median deviation to emphasize *efficiency differentials*
    energy_norm = robust_minmax_normalize(energy_density_centered)
    # Only promote high-efficiency tasks when safe (slack >= 0)
    energy_mask = (slack >= 0).astype(float)
    energy_efficiency_term = -energy_norm * energy_mask

    # === 3. Imminent latency risk: arctan-scaled volatility only in [0, 1.0) slack window ===
    base_latency = min_exec_time + min_comm_time + eps
    mean_base_latency = np.mean(base_latency) + eps
    relative_volatility = base_latency * (uncertainty + eps) / mean_base_latency
    # Arctan scaling bounded to [0, 1) for smooth, non-saturating risk amplification
    imminent_window = (slack >= 0) & (slack < 1.0)
    risk_arctan = np.where(imminent_window, 0.5 + (1.0 / np.pi) * np.arctan(relative_volatility), 0.0)
    latency_risk_term = robust_minmax_normalize(risk_arctan)

    # === 4. Fairness: clipped sqrt wait time, *only* for overdue tasks (slack < -0.1) ===
    aging_mask = (slack < -0.1).astype(float)
    aging_raw = np.where(aging_mask == 1, np.sqrt(np.clip(ready_wait_time, 0.0, 1e6)), 0.0)
    aging_norm = robust_minmax_normalize(aging_raw)
    # Weight by urgency magnitude to prioritize older tasks *more* when deadlines are tighter
    aging_term = -aging_norm * urgency_raw * aging_mask

    # === 5. Proximity penalty: linear penalty for tasks in (0, 1.0) slack window ===
    proximity_mask = (slack > 0) & (slack < 1.0)
    proximity_raw = np.where(proximity_mask, 1.0 - slack, 0.0)
    proximity_penalty = robust_minmax_normalize(proximity_raw)

    # === Final weighted score: smaller = higher priority ===
    score = (
        6.0 * deadline_score +
        2.5 * energy_efficiency_term +
        1.8 * latency_risk_term +
        1.1 * aging_term +
        0.9 * proximity_penalty
    )

    # Ensure finite, deterministic output
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    return score
