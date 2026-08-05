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
    v2 priority rule: Hard urgency dominance + critical-path energy gating + MAD-robust fairness +
    uncertainty-coupled slack sensitivity + adaptive work-density starvation relief.
    
    Key improvements:
    - Uses robust MAD normalization (from Parent 1) for outlier resilience, but with fallback to percentile-clipped min-max when MAD=0
    - Urgent tasks (slack <= 0) receive guaranteed minimum score (-1e15) for stronger hard-deadline enforcement
    - Energy penalty gated by *both* upward_rank > 75th-percentile AND slack < median_slack (tighter risk-aware gating)
    - Fairness term uses wait_per_work *and* normalized wait_time, activated only when remaining_work > median_work
    - Uncertainty coupling: scaled by inverse absolute slack *and* upward_rank, normalized robustly
    - All divisions guarded; NaN/inf replaced deterministically; no side effects
    """
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=float).copy()
    min_comm_time = np.asarray(min_comm_time, dtype=float).copy()
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float).copy()
    slack = np.asarray(slack, dtype=float).copy()
    upward_rank = np.asarray(upward_rank, dtype=float).copy()
    remaining_work = np.asarray(remaining_work, dtype=float).copy()
    ready_wait_time = np.asarray(ready_wait_time, dtype=float).copy()
    uncertainty = np.asarray(uncertainty, dtype=float).copy()
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)
    
    def robust_mad_norm(x):
        if x.size == 0:
            return np.zeros_like(x)
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        if mad < eps:
            # Fallback to percentile-clipped min-max when MAD degenerates
            p01 = np.percentile(x, 1.0)
            p99 = np.percentile(x, 99.0)
            x_clipped = np.clip(x, p01, p99)
            x_min = np.min(x_clipped)
            x_max = np.max(x_clipped)
            if x_max - x_min < eps:
                return np.zeros_like(x)
            return (x_clipped - x_min) / (x_max - x_min + eps)
        scale = mad * 3.0 + eps
        normed = (x - med) / scale
        return np.clip(normed, -3.0, 3.0)
    
    # Hard urgency: guaranteed top priority for overdue or at-risk tasks
    is_urgent = (slack <= 0.0).astype(float)
    
    # Robust inverse slack urgency: avoid division by zero and instability near zero
    abs_slack = np.abs(slack) + eps
    inv_slack = np.divide(1.0, abs_slack, out=np.zeros_like(slack), where=abs_slack != 0)
    inv_slack = np.clip(inv_slack, 0.1, 100.0)
    norm_inv_slack = robust_mad_norm(inv_slack)
    
    # Critical path importance: upward_rank weighted by uncertainty, gated by slack pressure
    median_slack = np.median(slack) if N > 1 else slack[0]
    slack_pressure_mask = (slack < median_slack).astype(float)
    risk_adjusted_rank = upward_rank * (1.0 + uncertainty)
    norm_risk_rank = robust_mad_norm(risk_adjusted_rank) * slack_pressure_mask
    
    # Duration and risk-adjusted duration
    duration = min_exec_time + min_comm_time + eps
    risk_duration = duration * (1.0 + uncertainty)
    
    # Energy density: marginal energy per risk-adjusted duration
    energy_density = np.divide(min_incremental_energy, risk_duration, out=np.zeros_like(min_incremental_energy), where=risk_duration != 0)
    norm_energy_density = robust_mad_norm(energy_density)
    
    # Energy penalty only applied to high-criticality *and* tight-slack tasks
    rank_threshold = np.percentile(upward_rank, 75.0) + eps
    high_rank_mask = (upward_rank > rank_threshold).astype(float)
    energy_penalty_mask = high_rank_mask * slack_pressure_mask
    critical_energy_penalty = norm_energy_density * norm_risk_rank * energy_penalty_mask
    
    # Fairness: starvation relief via wait-time per work, activated only for high-work-density tasks
    wait_per_work = np.divide(ready_wait_time, remaining_work + eps, out=np.zeros_like(ready_wait_time), where=remaining_work + eps != 0)
    median_work = np.median(remaining_work) if N > 1 else remaining_work[0]
    work_density_gate = (remaining_work > median_work).astype(float)
    norm_wait_time = robust_mad_norm(ready_wait_time)
    fairness_term = norm_wait_time * wait_per_work * work_density_gate
    
    # Uncertainty coupling: boosts urgency for high-rank, high-uncertainty, low-slack tasks
    uncertainty_boost = uncertainty * norm_inv_slack * norm_risk_rank
    norm_uncertainty_boost = robust_mad_norm(uncertainty_boost)
    
    # Final score composition with adaptive weights summing to 1.0
    score = np.full(N, 0.0, dtype=float)
    # Urgent tasks get absolute highest priority
    score = np.where(is_urgent, -1000000000000000.0, score)
    # Non-urgent tasks: weighted combination
    score = np.where(
        is_urgent,
        score,
        0.43 * norm_inv_slack + 
        0.29 * critical_energy_penalty + 
        0.16 * fairness_term + 
        0.08 * norm_uncertainty_boost + 
        0.04 * robust_mad_norm(remaining_work)
    )
    
    # Final sanitization
    score = np.nan_to_num(score, nan=1000000000000000.0, posinf=1000000000000000.0, neginf=-1000000000000000.0)
    score = np.clip(score, -1000000000000000.0, 1000000000000000.0)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
