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
    v2: Orthogonal, robust, and deadline-first priority with critical-energy density gating.
    
    Key improvements:
    - Urgency: arctan-based bounded risk (numerically stable near zero slack) → dominant when slack < 0
    - Criticality: upward_rank * remaining_work / (min_exec_time + eps), un-gated for stability & interpretability
    - Energy efficiency: min_incremental_energy per critical work unit, *only active when slack >= 0*
    - Fairness: sqrt(wait) normalized robustly → prevents starvation without overwhelming urgency
    - Risk: additive, bounded uncertainty penalty scaled by percentile, never multiplicative
    - All normalizations use IQR fallback with degenerate handling for N=1 or flat distributions
    - Strict term orthogonality: no overlapping slack modulation; slack only gates energy & scales urgency
    - Final score = weighted sum; smaller = higher priority; fully deterministic & finite-valued
    """
    eps = 1e-08
    # Ensure float arrays and avoid in-place mutation
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    # Robust IQR-based normalization with N=1 and flat-distribution fallback
    def robust_normalize(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        q25, q50, q75 = np.percentile(x, [25, 50, 75], axis=0, keepdims=False)
        iqr = q75 - q25
        scale = np.where(iqr > eps, iqr, 1.0)
        return (x - q50) / (scale + eps)

    # 1. Urgency: arctan-transformed slack — smooth, bounded [0,1], monotonic, stable at zero
    # Maps slack → 0 when slack → -∞, → 1 when slack → +∞, centered at slack=0
    arctan_urgency = (np.arctan(slack / 2.0) + np.pi / 2) / np.pi
    norm_urgency = robust_normalize(arctan_urgency)
    urgency_term = -3.0 * norm_urgency  # Higher urgency → lower score

    # 2. Criticality: raw importance per unit execution effort (MI/sec), un-gated for stability
    exec_effort = np.maximum(min_exec_time, eps)
    critical_density = upward_rank * (remaining_work / exec_effort)
    norm_critical = robust_normalize(critical_density)
    critical_term = -1.2 * norm_critical  # Higher criticality → lower score

    # 3. Energy efficiency: marginal energy per unit critical work — ONLY when slack >= 0
    # Encourages low-energy assignment *only* when deadline safety permits
    energy_per_crit = min_incremental_energy / (critical_density + eps)
    energy_gated = np.where(slack >= 0, energy_per_crit, 0.0)
    norm_energy = robust_normalize(energy_gated)
    efficiency_term = -0.8 * norm_energy  # Lower energy-per-crit → lower score

    # 4. Fairness: sqrt(wait) to penalize long queueing, clipped & normalized
    sqrt_wait = np.sqrt(np.maximum(ready_wait_time, 0.0) + eps)
    clipped_wait = np.clip(sqrt_wait, 0.0, np.percentile(sqrt_wait, 95) + eps)
    norm_wait = robust_normalize(clipped_wait)
    fairness_term = -0.25 * np.clip(norm_wait, -1.0, 1.5)  # Mild boost for aged tasks

    # 5. Uncertainty risk: additive penalty, bounded, uniform scaling
    unc_q75 = np.percentile(uncertainty, 75) + eps
    norm_uncertainty = np.clip(uncertainty / unc_q75, 0.0, 3.0)
    uncertainty_term = 0.4 * norm_uncertainty  # Higher risk → higher score

    # Combine all orthogonal terms
    score = urgency_term + critical_term + efficiency_term + fairness_term + uncertainty_term

    # Final sanitization: ensure finite, deterministic output
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    return score
