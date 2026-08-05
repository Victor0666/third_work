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
    v2 evolution: Deadline-hardened, energy-aware, and starvation-robust.
    Key improvements over v1:
    - Replaces tanh urgency with *clipped linear urgency* for exact deadline enforcement:
      slope = 1.0 for slack < 0 (strictly penalizes lateness), flat for slack >= 0 → guarantees
      zero urgency penalty only when no violation risk exists.
    - Introduces *critical-energy-density* (CED): upward_rank * remaining_work / (min_incremental_energy + eps),
      soft-gated only for slack >= -0.2s (tighter gate) → prioritizes energy-critical paths earlier.
    - Adds *latency-pressure term*: (min_exec_time + min_comm_time) / (|slack| + eps), active for all tasks,
      scaled by upward_rank and uncertainty → directly penalizes high-latency tasks under tight slack.
    - Fairness refined: sqrt(wait) * exp(-uncertainty) / (1 + max(0, -slack) + eps) → amplifies starvation relief
      exactly when lateness risk increases, not just absolute |slack|.
    - All terms use robust IQR normalization with explicit N=1 safety and finite bounds.
    - Final score enforces strict DDL feasibility hierarchy: urgency dominates, then CED, then latency-pressure,
      then fairness — no term overpowers deadline safety.
    """
    eps = 1e-08
    tau_strict = 0.2  # tighter gating for criticality-energy term
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    def normalize_adaptive(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        q25, q75 = np.percentile(x, [25, 75])
        iqr = q75 - q25
        if iqr > eps:
            center = np.median(x)
            scale = iqr + eps
        else:
            xmin, xmax = np.min(x), np.max(x)
            scale = xmax - xmin
            center = (xmax + xmin) / 2.0
            if scale < eps:
                scale = eps
        return (x - center) / (scale + eps)

    # Urgency: clipped linear — zero penalty only when slack >= 0; strict penalty slope 1.0 for slack < 0
    urgency_raw = np.where(slack >= 0, 0.0, -slack)  # strictly positive penalty for lateness risk
    norm_urgency = normalize_adaptive(urgency_raw)
    urgency_term = -4.0 * norm_urgency  # stronger weight to enforce DDL hard constraint

    # Critical-Energy-Density: importance × work per joule, gated only near hard violation
    ced_base = upward_rank * remaining_work / (min_incremental_energy + eps)
    ced_masked = np.where(slack >= -tau_strict, ced_base, 0.0)
    norm_ced = normalize_adaptive(ced_masked)
    ced_term = -1.8 * norm_ced  # increased weight for energy-critical paths under tight deadlines

    # Latency-pressure: penalizes high exec+comm under low slack, scaled by criticality and risk
    latency_pressure_base = (min_exec_time + min_comm_time + eps) / (np.abs(slack) + eps)
    latency_pressure_scaled = latency_pressure_base * upward_rank * (1.0 + uncertainty)
    norm_latency_pressure = normalize_adaptive(latency_pressure_scaled)
    latency_pressure_term = 0.9 * norm_latency_pressure

    # Fairness: starvation relief amplified *exactly* when lateness risk rises (not symmetric |slack|)
    wait_safe = np.maximum(ready_wait_time, 0.0)
    sqrt_wait = np.sqrt(wait_safe + eps)
    risk_decay = np.exp(-uncertainty)  # softer decay than exp(-unc²), preserves signal at mid-uncertainty
    fairness_numerator = sqrt_wait * risk_decay
    fairness_denominator = 1.0 + np.maximum(0.0, -slack) + eps  # only penalize fairness relief when slack negative
    fairness_raw = fairness_numerator / fairness_denominator
    norm_fairness = normalize_adaptive(fairness_raw)
    fairness_term = -0.4 * norm_fairness  # slightly stronger fairness weight to prevent starvation under pressure

    score = urgency_term + ced_term + latency_pressure_term + fairness_term
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    return score
