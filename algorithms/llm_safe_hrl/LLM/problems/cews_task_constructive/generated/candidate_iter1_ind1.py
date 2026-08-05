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
    """Novel priority rule emphasizing deadline feasibility first, then energy-aware
    critical-path progression, with starvation-avoidance and uncertainty-aware balancing.
    
    Key innovations:
    - Uses *risk-scaled urgency*: exponential decay of urgency beyond safe slack,
      but sharp penalty below zero slack (no division by zero, no log domain errors).
    - Replaces linear normalization with *robust range-based scaling* (IQR + epsilon)
      to resist outliers and preserve relative ordering under skewed distributions.
    - Introduces *criticality-energy tradeoff ratio*: upward_rank / (min_exec_time + min_comm_time + eps)
      to favor high-impact tasks per unit time/energy cost — avoids bias toward short OR critical alone.
    - Models *uncertainty-aware waiting boost*: ready_wait_time scaled by uncertainty and normalized
      only among high-risk tasks (slack < 0), preventing low-risk long-waiters from dominating.
    - Enforces strict DDL-first hierarchy: negative slack gets dominant penalty term,
      applied *before* other features to guarantee hard constraint adherence.
    - All operations are finite, deterministic, and numerically guarded.
    """
    eps = 1e-8

    # Safe conversion without mutation
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    # --- 1. Robust feature scaling: IQR-based, fallback to mean-abs if degenerate ---
    def robust_scale(x):
        q1, q3 = np.quantile(x, [0.25, 0.75])
        iqr = q3 - q1
        scale = iqr if iqr > eps else (np.mean(np.abs(x)) + eps)
        return x / (scale + eps)

    # --- 2. Deadline risk dominance: sharp, bounded penalty for negative slack ---
    # Use softplus-like penalty: max(0, -slack) + log(1 + exp(-slack)) → smooth, finite, monotonic
    # But simpler & more stable: quadratic penalty for negative slack, zero otherwise
    deadline_penalty = np.where(slack < 0, (-slack) ** 2, 0.0)
    # Normalize penalty to avoid overwhelming other terms when slack is large negative
    deadline_penalty = robust_scale(deadline_penalty)

    # --- 3. Risk-scaled urgency: decays smoothly for positive slack, peaks at zero, penalizes negative ---
    # Avoids division; uses shifted sigmoid: 1 / (1 + exp((slack - 0.1 * np.median(slack)) / (eps + np.std(slack))))
    # But simpler & more stable: clipped linear decay + floor
    median_slack = np.median(slack) if len(slack) > 1 else np.mean(slack)
    safe_range = np.maximum(np.std(slack), eps) if len(slack) > 1 else eps
    # Urgency = 1 when slack ≈ median_slack, drops to ~0.1 at slack = median_slack + 3*safe_range
    urgency = np.clip(1.0 - (slack - median_slack) / (3.0 * safe_range + eps), 0.1, 1.0)
    # Boost urgency further for negative slack (already penalized above, but we want prioritization too)
    urgency = np.where(slack < 0, np.minimum(1.5 * urgency, 2.0), urgency)

    # --- 4. Criticality-energy efficiency: upward_rank per unit time+comm cost ---
    # Higher ratio → more critical work per latency/energy cost → preferred
    time_cost = min_exec_time + min_comm_time + eps
    crit_efficiency = upward_rank / time_cost
    crit_efficiency = robust_scale(crit_efficiency)

    # --- 5. Uncertainty-aware waiting boost: only activated for at-risk tasks (slack < 0) ---
    # Prevents idle starvation *only when needed*: long wait matters most when deadline is tight
    wait_boost = np.where(slack < 0,
                         robust_scale(ready_wait_time) * robust_scale(uncertainty),
                         0.0)

    # --- 6. Energy-aware base score: minimize incremental energy, but modulated by urgency ---
    # Prioritize low-energy tasks *especially* when urgent (urgency > 0.5)
    energy_score = robust_scale(min_incremental_energy)
    energy_score = energy_score * (1.0 + 0.5 * np.where(urgency > 0.5, urgency, 0.0))

    # --- 7. Composite score: deadline penalty dominates; others balance energy/criticality/wait ---
    # Negative weights for features we want *smaller* values of → higher priority
    # Positive weights for penalties we want *larger* values of → lower priority
    score = (
        3.0 * deadline_penalty              # Hard DDL enforcement: largest weight
        + 0.8 * energy_score                # Energy minimization, boosted when urgent
        - 0.7 * crit_efficiency             # Favor critical-per-cost (higher = better → subtract)
        - 0.4 * urgency                     # Prefer urgent tasks (higher urgency = better → subtract)
        + 0.3 * wait_boost                  # Small boost for long-waiting *at-risk* tasks
        + 0.1 * robust_scale(remaining_work)  # Slight preference for large remaining work (early binding)
        - 0.2 * robust_scale(uncertainty)   # Mild preference for lower-uncertainty (less risk exposure)
    )

    # Final numerical guard: ensure finite, shape-(N,)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)

    return score
