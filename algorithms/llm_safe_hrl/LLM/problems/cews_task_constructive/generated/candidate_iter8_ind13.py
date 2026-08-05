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
    Hybrid priority rule combining v1's stability & v2's deadline-gated energy optimization.
    Key innovations:
    - Bounded arctan urgency (v2) + robust slack penalty scaling (v1-inspired) for smoother deadline response
    - Critical-energy density (CED) gated by slack >= 0 (v2), but enhanced with latency-aware normalization
    - Unified uncertainty handling: inflates latency only when slack > 0 AND uncertainty > eps, avoiding over-penalization
    - Aging term uses clipped sqrt wait *modulated by urgency* to boost long-waiting tasks only under deadline pressure
    - All normalizations use min-max with IQR fallback (v2) + explicit N=1 safety (v1)
    - Strict term ordering: deadline dominates (weight 4.0), CED second (2.5), efficiency third (1.0), aging fourth (0.3)
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

    # === Deadline urgency: arctan-bounded, smooth, and monotonic ===
    # Maps slack → [0,1]: 0 when slack → -∞ (critical), 1 when slack → +∞ (relaxed)
    arctan_urgency = (np.arctan(-slack / 2.0) + np.pi / 2) / np.pi
    urgency_term = 4.0 * arctan_urgency

    # === Critical-energy density (CED): impact per joule, only active when slack >= 0 ===
    base_ced = (upward_rank * remaining_work) / (min_incremental_energy + eps)
    ced_masked = np.where(slack >= 0, base_ced, 0.0)

    # Robust min-max normalization with IQR fallback and N=1 safety
    def robust_minmax(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        rng = np.max(x) - np.min(x)
        if rng > eps:
            return (x - np.min(x)) / (rng + eps)
        else:
            q75, q25 = np.percentile(x, [75, 25])
            iqr = q75 - q25
            if iqr > eps:
                return (x - np.median(x)) / (iqr + eps)
            else:
                return np.zeros_like(x)

    norm_ced = robust_minmax(ced_masked)
    critical_energy_term = -2.5 * norm_ced  # Higher weight: favors high-impact/low-energy tasks when feasible

    # === Efficiency term: energy-per-latency ratio, also slack-gated ===
    base_latency = min_exec_time + min_comm_time + eps
    # Uncertainty inflates latency only when slack > 0 AND uncertainty is non-negligible
    inflation_factor = np.where((slack > 0) & (uncertainty > eps), 
                               np.clip(uncertainty, 0.0, 2.0), 0.0)
    inflated_latency = base_latency * (1.0 + 0.1 * inflation_factor)
    energy_per_latency = min_incremental_energy / (inflated_latency + eps)
    eff_ratio_gated = np.where(slack >= 0, energy_per_latency, 0.0)
    norm_eff_ratio = robust_minmax(eff_ratio_gated)
    efficiency_term = 1.0 * norm_eff_ratio  # Reward low-energy-per-latency when feasible

    # === Aging term: sqrt wait time, boosted only under urgency (slack < 0) ===
    sqrt_wait = np.sqrt(np.maximum(ready_wait_time, 0.0) + eps)
    # Clip at 95th percentile to prevent outlier dominance
    wait_cap = np.percentile(sqrt_wait, 95) + eps if sqrt_wait.size > 1 else np.max(sqrt_wait) + eps
    clipped_wait = np.clip(sqrt_wait, 0.0, wait_cap)
    norm_wait = robust_minmax(clipped_wait)
    # Modulate aging boost by urgency: stronger effect when deadlines are tight
    aging_boost = np.where(slack < 0, 1.0 + np.abs(slack) / (np.abs(slack).max() + eps), 1.0)
    aging_term = -0.3 * norm_wait * aging_boost

    # Combine all terms — smaller score = higher priority
    score = urgency_term + critical_energy_term + efficiency_term + aging_term

    # Final sanitization: ensure finite, deterministic output
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    
    # Enforce shape (N,) — critical for single-task case
    return score.reshape(-1)
