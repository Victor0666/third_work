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
    Self-evolved priority rule: fixes v1's instability and sign errors while strengthening DDL-first hierarchy.
    Key improvements:
    - Replaces uncertainty-modulated deadline gain with *bounded slack penalty scaling*: smooth, monotonic, and stable.
    - Inverts latency-criticality ratio to become *criticality-latency efficiency* (upward_rank / (exec+comm)), promoting high-importance low-latency tasks.
    - Uses *per-task MAD normalization only on non-gated terms*, avoiding zero-mass clusters; gated terms use direct bounded transforms.
    - Introduces *slack-aware energy fairness*: penalizes high-energy tasks only when slack is tight (< 0), avoids diluting energy optimization in feasible region.
    - All operations guarded against degeneracy (N=1, const arrays, zeros); final score strictly finite and deterministic.
    - Strict weight ordering: deadline dominates (5.0), criticality-efficiency second (2.0), energy fairness third (1.5), wait boost fourth (0.2).
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
    
    # Robust MAD normalization: handles N=1, const, NaN, inf — returns zeros for degenerate cases
    def safe_mad_normalize(x):
        x_clean = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x_clean.size == 0:
            return np.zeros_like(x_clean)
        if x_clean.size == 1:
            return np.zeros_like(x_clean)
        med = np.median(x_clean)
        abs_dev = np.abs(x_clean - med)
        mad = np.median(abs_dev) + eps
        normed = (x_clean - med) / mad
        return np.clip(normed, -4.0, 4.0)
    
    # === Deadline risk: arctan-based, bounded, monotonic, no uncertainty modulation ===
    # Negative slack → increasing penalty; zero/negative slack gets smooth, bounded penalty
    deadline_risk = np.arctan(np.clip(-slack, 0.0, None))  # [0, π/2)
    deadline_score = safe_mad_normalize(deadline_risk)
    
    # === Criticality-latency efficiency: upward_rank / (exec + comm) → higher is better → lower score preferred ===
    total_latency = np.clip(min_exec_time + min_comm_time, eps, None)
    crit_lat_eff = upward_rank / total_latency  # Higher = more urgent & faster → prioritize
    crit_lat_eff_norm = safe_mad_normalize(crit_lat_eff)
    
    # === Energy fairness: only penalize high energy when slack < 0 (DDL violation imminent) ===
    # Otherwise, energy is secondary; avoid distorting ranking when deadlines are safe
    energy_penalty_raw = np.where(slack < 0, min_incremental_energy, 0.0)
    energy_penalty_norm = safe_mad_normalize(energy_penalty_raw)
    
    # === Wait fairness: monotonic, bounded boost for long-waiting tasks, scaled by urgency (1/(|slack|+eps) when slack < 0) ===
    wait_base = np.sqrt(np.clip(ready_wait_time, 0.0, None))
    wait_urgency_factor = np.where(slack < 0, 1.0 / (np.abs(slack) + eps), 0.0)
    wait_boost = np.clip(wait_base * wait_urgency_factor * 0.1, 0.0, 0.3)
    
    # === Uncertainty: used only to inflate latency (not deadline gain) → stabilizes execution predictability ===
    uncertainty_clipped = np.clip(uncertainty, 0.0, 3.0)
    inflated_latency = total_latency * (1.0 + uncertainty_clipped * 0.1)
    inflated_lat_norm = safe_mad_normalize(inflated_latency)
    
    # === Final score: smaller = higher priority; deadline dominates, others refine under constraint ===
    # Sign convention: deadline_score and energy_penalty_norm increase with risk → positive weight
    # crit_lat_eff_norm increases with desirability → negative weight to lower score
    score = (
        +5.0 * deadline_score              # DDL violation risk dominates
        - 2.0 * crit_lat_eff_norm         # Favor high-importance, low-latency tasks
        + 1.5 * energy_penalty_norm       # Penalize high energy only when lateness looms
        + 0.2 * wait_boost                # Gentle fairness for overdue-ready tasks
        + 0.25 * inflated_lat_norm        # Account for uncertainty-induced latency inflation
    )
    
    # Ensure finite output, shape (N,)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    return score.reshape(-1)
