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
    """Novel priority rule emphasizing deadline-criticality-aware energy efficiency.

    Key innovations:
    - Replaces linear normalization with robust percentile-based scaling to resist outliers.
    - Introduces 'risk-adjusted urgency': combines normalized negative slack * (1 + uncertainty)
      only when slack < 0, else zero — avoids diluting urgency for truly critical tasks.
    - Uses energy-efficiency ratio: min_incremental_energy / (min_exec_time + min_comm_time + eps)
      to favor energy-per-second efficiency, scaled robustly.
    - Adds 'criticality density': upward_rank * remaining_work / (max(remaining_work) + eps),
      highlighting high-impact, high-work tasks on critical paths.
    - Introduces starvation relief via sigmoid-transformed ready_wait_time to cap benefit
      and prevent long-waiting tasks from dominating.
    - All components are sign-consistent: lower score = higher priority; negative slack yields
      strong negative contribution (i.e., high priority).
    """
    eps = 1e-8

    # Safe array conversion without in-place mutation
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    # Robust percentile-based normalization: scale by IQR + eps to resist outliers
    def robust_normalize(x):
        q75, q25 = np.percentile(x, 75), np.percentile(x, 25)
        iqr = q75 - q25 + eps
        center = np.median(x)
        return (x - center) / iqr

    # === 1. Deadline Risk Priority (strongest driver) ===
    # Only activate penalty for negative slack; magnitude scales with uncertainty
    deadline_risk_mask = (slack < 0).astype(float)
    risk_magnitude = -slack * deadline_risk_mask * (1.0 + uncertainty)  # larger = more urgent
    # Normalize risk magnitude robustly; negative contribution → lowers score
    norm_risk = -robust_normalize(risk_magnitude)  # now smaller value = more urgent

    # === 2. Energy Efficiency Ratio ===
    # Favor low marginal energy per unit time (J/s), but avoid division by near-zero
    time_sum = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / time_sum
    norm_efficiency = robust_normalize(energy_per_sec)

    # === 3. Criticality Density ===
    # Upward rank × relative remaining work → highlights bottlenecks with high descendant load
    max_rw = np.max(remaining_work) + eps
    critical_density = upward_rank * (remaining_work / max_rw)
    norm_critical = robust_normalize(critical_density)

    # === 4. Starvation Relief (bounded, smooth) ===
    # Sigmoid of normalized wait time: saturates at ~0.9 to prevent dominance
    wait_norm = (ready_wait_time - np.mean(ready_wait_time)) / (np.std(ready_wait_time) + eps)
    starvation_relief = 1.0 / (1.0 + np.exp(-0.5 * wait_norm))  # [0,1]
    # Convert to priority boost: subtract (since lower score = better)
    norm_starvation = -robust_normalize(starvation_relief)

    # === 5. Uncertainty-Aware Communication Penalty ===
    # Penalize high comm time * uncertainty only when slack is tight (|slack| < median |slack|)
    abs_slack = np.abs(slack)
    tightness_mask = (abs_slack < (np.median(abs_slack) + eps)).astype(float)
    comm_risk_penalty = min_comm_time * uncertainty * tightness_mask
    norm_comm_risk = robust_normalize(comm_risk_penalty)

    # === Final score: weighted sum with sign-consistent contributions ===
    # All terms designed so smaller = better (negative terms pull score down for urgent/efficient tasks)
    score = (
        0.30 * norm_efficiency           # lower energy/sec → lower score
        + 0.25 * norm_critical          # higher critical density → higher score → penalized
        + 0.20 * norm_comm_risk         # higher comm+uncertainty under tight slack → higher score
        + 0.10 * robust_normalize(min_exec_time)  # longer exec → higher score
        + 0.05 * robust_normalize(uncertainty)    # higher uncertainty alone → mild penalty
        + norm_risk                     # strongest term: negative slack + uncertainty → strongly lowers score
        + norm_starvation               # longer wait → lowers score (relief)
    )

    # Final numerical safeguard
    return np.nan_to_num(
        score,
        nan=1e12,
        posinf=1e12,
        neginf=-1e12
    )
