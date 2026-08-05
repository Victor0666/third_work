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
    Hybrid priority rule combining hard deadline gating (v1) with robust risk-aware normalization (v0).
    Key innovations:
    - Hard deadline gating: zero energy term for slack < 0, ensuring DDL-violating tasks dominate.
    - Bounded arctan-based urgency (v0) replaces exponential to avoid overflow and improve numerical stability near zero.
    - Critical-energy density (upward_rank * remaining_work / energy) gated by slack >= 0, normalized via MAD for outlier resilience.
    - Latency-risk ratio (exec+comm)*uncertainty/(|slack|+eps) scaled by clipped sigmoid of (-slack) to emphasize high-risk tasks.
    - Aging only activates under deadline pressure (slack < 0) and is normalized robustly (MAD + fallback).
    - All normalizations use unified safe_mad_normalize with N=1 and constant-array fallbacks.
    - Final weights prioritize deadline safety (4.0), critical efficiency (2.5), latency risk (1.8), aging (0.7), uncertainty (0.5).
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
        """Robust MAD normalization: handles N=1, constant arrays, and degeneracy."""
        if x.size == 1:
            return np.zeros_like(x)
        median_x = np.median(x)
        mad = np.median(np.abs(x - median_x)) + eps
        if mad < eps:
            return np.zeros_like(x)
        normed = (x - median_x) / mad
        return np.clip(normed, -3.0, 3.0)

    # Deadline urgency: bounded arctan(-slack) → smooth, monotonic, stable at extremes
    deadline_urgency_raw = np.arctan(-slack) + np.pi / 2  # [0, π], maps slack→∞ → 0, slack→-∞ → π
    deadline_urgency = deadline_urgency_raw / np.pi  # [0, 1]
    deadline_score = safe_mad_normalize(deadline_urgency)

    # Energy efficiency term: only active when slack >= 0 (hard gating)
    energy_mask = (slack >= 0).astype(float)
    critical_energy_density = upward_rank * (remaining_work + eps) / (min_incremental_energy + eps)
    critical_energy_norm = safe_mad_normalize(critical_energy_density)
    energy_efficiency_term = -critical_energy_norm * energy_mask

    # Latency-risk ratio: penalizes volatile long tasks near deadline
    base_latency = min_exec_time + min_comm_time + eps
    latency_risk_ratio = base_latency * (uncertainty + eps) / (np.abs(slack) + eps)
    # Scale by urgency: higher penalty when slack is negative (via clipped sigmoid of -slack)
    urgency_sigmoid = 1.0 / (1.0 + np.exp(-np.clip(-slack, -10.0, 10.0)))
    latency_risk_scaled = latency_risk_ratio * urgency_sigmoid
    latency_term = safe_mad_normalize(latency_risk_scaled)

    # Aging term: only applied under deadline pressure (slack < 0), normalized robustly
    aging_term_raw = np.where(slack < 0, ready_wait_time, 0.0)
    aging_term = -safe_mad_normalize(aging_term_raw) * np.clip(deadline_urgency, 0.0, 1.0)

    # Uncertainty term: only active when slack > 0 (risk-aware modulation)
    uncertainty_gated = np.where(slack > 0, uncertainty, 0.0)
    uncertainty_norm = safe_mad_normalize(uncertainty_gated)

    # Final weighted score: smaller = higher priority
    score = (
        4.0 * deadline_score +
        2.5 * energy_efficiency_term +
        1.8 * latency_term +
        0.7 * aging_term +
        0.5 * uncertainty_norm
    )

    # Ensure finite output, deterministic, shape (N,)
    return np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
