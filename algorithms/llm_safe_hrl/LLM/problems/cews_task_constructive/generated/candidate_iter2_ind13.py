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
    Hybrid priority rule combining deadline safety, critical-path density, energy-latency efficiency,
    starvation mitigation, and uncertainty-aware normalization — with robust scale-invariant treatment.
    
    Key improvements over v1:
    - Replaces linear wait penalty with sqrt-scaled anti-starvation boost (like v0) but sign-corrected:
      longer wait → *lower* score (higher priority), bounded and normalized.
    - Introduces uncertainty-gated energy signal: high uncertainty suppresses energy penalty,
      favoring predictable low-energy VMs (v0 insight) while preserving v1's ratio structure.
    - Uses clipped slack-based urgency *and* near-deadline boost (v0) for finer-grained deadline sensitivity.
    - Critical work density enhanced with uncertainty damping: high-uncertainty tasks get reduced weight
      on descendant work to avoid over-committing to risky estimates.
    - All features normalized via IQR + median, with explicit zero-variance fallback and epsilon safety.
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
    
    def robust_iqr_normalize(x):
        """IQR-normalize: (x - median) / (IQR + eps); fallback to mean-abs if IQR ~ 0."""
        q25, q50, q75 = np.percentile(x, [25, 50, 75], axis=0, keepdims=False)
        iqr = q75 - q25
        scale = iqr if iqr > eps else np.mean(np.abs(x - q50)) + eps
        return (x - q50) / (scale + eps)
    
    # Deadline urgency: strong penalty for negative slack, moderate boost for near-deadline (0 <= slack < 2*med_exec)
    med_exec = np.median(min_exec_time) + eps
    urgent_risk = np.where(slack < 0, -np.clip(-slack, 0, 100.0), 0.0)
    near_deadline = np.where((slack >= 0) & (slack < 2 * med_exec), 1.0, 0.0)
    deadline_urgency = urgent_risk + 0.3 * near_deadline * med_exec
    
    # Critical work density: upward_rank * (remaining_work / exec_effort), damped by uncertainty
    exec_effort = np.maximum(min_exec_time, eps)
    unc_damp = 1.0 / (1.0 + uncertainty)  # high uncertainty → lower weight on descendant work
    work_density = upward_rank * (remaining_work / exec_effort) * unc_damp
    
    # Energy-latency efficiency: energy per total latency cost, uncertainty-gated
    latency_cost = min_exec_time + min_comm_time + eps
    base_energy_latency = min_incremental_energy / latency_cost
    # High uncertainty reduces penalty magnitude: prefer predictable low-energy over risky low-energy
    energy_latency_score = base_energy_latency * (1.0 / (1.0 + uncertainty))
    
    # Anti-starvation: sqrt-scaled wait time → *lower* score for longer waits (higher priority)
    wait_boost = -np.sqrt(ready_wait_time + eps)  # negative sign: longer wait → higher priority
    
    # Uncertainty term: penalize high uncertainty to encourage early scheduling of risky tasks
    # but bounded to avoid domination
    uncertainty_term = np.clip(uncertainty, 0.0, 10.0)
    
    # Normalize each component
    norm_deadline = robust_iqr_normalize(deadline_urgency)
    norm_work_density = robust_iqr_normalize(work_density)
    norm_energy_latency = robust_iqr_normalize(energy_latency_score)
    norm_wait = robust_iqr_normalize(wait_boost)
    norm_uncertainty = robust_iqr_normalize(uncertainty_term)
    
    # Final weighted score: smaller = higher priority
    # Weights sum to 1.0; deadline and criticality dominate, energy-efficiency secondary, wait/uncertainty minor
    score = (
        0.35 * norm_deadline           # strongest weight: hard DDL safety
        - 0.25 * norm_work_density     # promote critical & heavy tasks (sign flipped: lower score = higher priority)
        - 0.20 * norm_energy_latency   # reward energy-efficient execution per latency unit
        + 0.10 * norm_wait             # anti-starvation: longer wait → lower score → higher priority
        + 0.10 * norm_uncertainty      # mild penalty for high uncertainty to reduce risk accumulation
    )
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    return score
