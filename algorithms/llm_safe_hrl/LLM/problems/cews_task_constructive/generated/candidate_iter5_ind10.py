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
    Hybrid priority rule: merges v0's energy-per-work focus and smooth deadline gating
    with v1's robust MAD normalization, arctan urgency, and uncertainty-weighted fairness.
    
    Key improvements:
    - Uses arctan(-slack) for monotonic, bounded deadline urgency (v1) but scales it
      via sigmoid-derived risk factor from v0 for smoother near-deadline response.
    - Adopts v0's energy_per_work = min_incremental_energy / (remaining_work + eps)
      as primary energy signal — directly aligns with DDL-safe energy minimization.
    - Keeps v1's deterministic MAD-based robust_normalize (no percentile, works for N=1).
    - Introduces *slack-gated criticality*: applies upward_rank / (remaining_work + eps)
      only when slack >= 0, prioritizing critical paths early while deferring them under pressure.
    - Uncertainty linearly amplifies both deadline risk and wait boost, but capped to avoid explosion.
    - Adds minimal communication-aware fairness: penalizes high min_comm_time only when slack < 0.
    - All weights sum to 1.0 and are tuned to enforce hard-DDL priority first, then energy.
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

    def robust_normalize(x):
        center = np.median(x)
        mad = np.median(np.abs(x - center)) + eps
        return (x - center) / (mad + eps)

    # Deadline urgency: symmetric arctan baseline, smoothed by sigmoid risk factor
    base_deadline_urgency = np.arctan(-slack)  # more negative slack → larger positive score
    # Sigmoid risk factor: peaks near deadline, bounded [0, 1], avoids infinity
    tau = np.clip(np.mean(np.abs(slack)) + eps, eps, 1000.0)
    risk_factor = 1.0 / (1.0 + np.exp(slack / tau + eps))
    deadline_score = base_deadline_urgency * (1.0 + np.clip(uncertainty, 0.0, 2.0)) * risk_factor

    # Energy efficiency: marginal energy per unit remaining work — favors low-energy tasks per MI
    energy_per_work = min_incremental_energy / (remaining_work + eps)
    energy_score = robust_normalize(energy_per_work)

    # Criticality: only active when slack >= 0 (safe window), else suppressed to prioritize urgency
    critical_mask = (slack >= 0.0).astype(float)
    normalized_criticality = upward_rank / (remaining_work + eps)
    critical_score = -robust_normalize(normalized_criticality) * critical_mask

    # Starvation relief: tanh-scaled wait time, uncertainty-amplified, active even under pressure
    median_wait = np.median(ready_wait_time) + eps
    wait_base = np.tanh(ready_wait_time / median_wait)
    wait_boost = wait_base * (1.0 + np.clip(uncertainty, 0.0, 2.0))
    wait_score = -robust_normalize(wait_boost)

    # Communication fairness: penalize high comm time only when slack < 0 (deadline pressure)
    comm_mask = (slack < 0.0).astype(float)
    comm_score = robust_normalize(min_comm_time) * comm_mask

    # Execution time fairness: mild penalty for long exec when slack < 0 to prevent starvation of short tasks
    exec_mask = (slack < 0.0).astype(float)
    exec_score = robust_normalize(min_exec_time) * exec_mask

    # Final weighted score: hard-DDL safety dominates; energy dominates in safe region
    # Weights sum to 1.0 and reflect priority hierarchy: deadline > criticality > energy > fairness
    score = (
        0.40 * deadline_score +
        0.25 * critical_score +
        0.20 * energy_score +
        0.08 * wait_score +
        0.05 * comm_score +
        0.02 * exec_score
    )

    return np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
