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
    v2 priority rule: Hard deadline lockstep + robust criticality gating + 
    uncertainty-coupled slack pressure + starvation-aware fairness + 
    strict monotonic urgency + outlier-resilient normalization.

    Key improvements:
    - Combines Parent 2's adaptive slack gating (10th percentile) and strict monotonic urgency
      with Parent 1's local slack pressure (quantile-based) and latency-uncertainty coupling
      *only for tight-slack tasks* to avoid diluting urgency dominance.
    - Uses robust_zscore (MAD-based) for all features — proven resilience to skew and outliers.
    - Introduces *urgency-modulated fairness*: wait_efficiency penalty activated only when slack > 0,
      preventing starvation under deadline pressure while preserving urgency ordering.
    - Replaces fragile energy-density ratio with *risk-normalized energy*: 
      min_incremental_energy / (duration * (1 + uncertainty)), guarding against zero/inf.
    - All weights sum to 1.0: urgency (0.43) > energy-gated (0.22) > criticality (0.15) > 
      fairness (0.10) > uncertainty_slack (0.06) > work_density (0.04).
    - Strict finite sanitization, no in-place modification, deterministic for identical inputs.
    """
    eps = 1e-08
    # Clean and convert all inputs to float64, no in-place mutation
    min_exec_time = np.asarray(min_exec_time, dtype=np.float64)
    min_comm_time = np.asarray(min_comm_time, dtype=np.float64)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slack = np.asarray(slack, dtype=np.float64)
    upward_rank = np.asarray(upward_rank, dtype=np.float64)
    remaining_work = np.asarray(remaining_work, dtype=np.float64)
    ready_wait_time = np.asarray(ready_wait_time, dtype=np.float64)
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    
    # Sanitize NaN/inf/neginf to zero
    inputs = [min_exec_time, min_comm_time, min_incremental_energy, slack, 
              upward_rank, remaining_work, ready_wait_time, uncertainty]
    cleaned = [np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0) for arr in inputs]
    min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, \
        remaining_work, ready_wait_time, uncertainty = cleaned
    
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=np.float64)
    
    # Robust z-score using MAD; handles small-N and degenerate cases
    def robust_zscore(x):
        if x.size == 0:
            return np.zeros_like(x)
        med = np.median(x)
        dev = x - med
        mad = np.median(np.abs(dev))
        if mad < eps:
            return np.zeros_like(x)
        z = dev / (mad + eps)
        return np.clip(z, -5.0, 5.0)
    
    # Urgency: hard dominance for overdue tasks, strict monotonic deficit for others
    is_urgent = (slack <= 0.0).astype(np.float64)
    urgency_score = np.full(N, -1000000000000000.0, dtype=np.float64)
    duration = min_exec_time + min_comm_time + eps
    slack_deficit = np.maximum(-slack, 0.0)
    norm_urgency = np.divide(slack_deficit, duration + eps, out=np.zeros_like(slack_deficit), where=duration + eps != 0)
    norm_urgency = np.clip(norm_urgency, 0.0, 1.0)  # strict monotonic [0,1]
    
    # Adaptive energy gating: only most urgent 10% of ready tasks trigger energy optimization
    if N > 1:
        slack_10p = np.percentile(slack, 10.0, method='midpoint')
        energy_gate = (slack < slack_10p).astype(np.float64)
    else:
        energy_gate = np.ones(N, dtype=np.float64)
    
    # Risk-normalized energy: penalize high energy per effective duration (incl. uncertainty)
    effective_duration = duration * (1.0 + uncertainty + eps)
    risk_norm_energy = np.divide(min_incremental_energy, effective_duration, 
                                 out=np.zeros_like(min_incremental_energy), 
                                 where=effective_duration != 0)
    risk_norm_energy = np.where(np.isfinite(risk_norm_energy), risk_norm_energy, 0.0)
    norm_energy = robust_zscore(risk_norm_energy)
    
    # Criticality: upward_rank scaled by normalized duration (progress velocity)
    progress_velocity = np.divide(upward_rank, duration, out=np.zeros_like(upward_rank), where=duration != 0)
    norm_progress_velocity = robust_zscore(progress_velocity)
    
    # Fairness: starvation-aware wait efficiency — only active when slack > 0
    wait_efficiency = np.divide(ready_wait_time, remaining_work + eps, 
                                out=np.zeros_like(ready_wait_time), 
                                where=remaining_work + eps != 0)
    wait_efficiency = np.where(np.isfinite(wait_efficiency), wait_efficiency, 0.0)
    norm_wait_efficiency = robust_zscore(wait_efficiency)
    fairness_mask = (slack > 0.0).astype(np.float64)
    
    # Uncertainty-slack sensitivity: relative uncertainty scaled by slack pressure
    abs_slack = np.abs(slack) + eps
    # Local slack pressure: quantile-based, robust to small-N
    slack_clean = np.nan_to_num(slack, nan=0.0, posinf=0.0, neginf=0.0)
    if N > 1:
        slack_q90 = np.percentile(slack_clean, 90)
        slack_q10 = np.percentile(slack_clean, 10)
        slack_range = max(slack_q90 - slack_q10, eps)
        slack_pressure = np.clip((slack_q90 - slack_clean) / slack_range, 0.0, 1.0)
    else:
        slack_pressure = np.ones_like(slack_clean)
    unc_sensitivity = uncertainty * slack_pressure
    norm_unc_sensitivity = robust_zscore(unc_sensitivity)
    
    # Work density penalty: energy per MI, only when not urgent
    energy_per_mi = np.divide(min_incremental_energy, remaining_work + eps, 
                             out=np.zeros_like(min_incremental_energy), 
                             where=remaining_work + eps != 0)
    energy_per_mi = np.where(np.isfinite(energy_per_mi), energy_per_mi, 0.0)
    norm_energy_per_mi = robust_zscore(energy_per_mi)
    work_density_mask = (slack > 0.0).astype(np.float64)
    
    # Weighted combination — urgency dominates unconditionally
    w_urgency = 0.43
    w_energy = 0.22
    w_critical = 0.15
    w_fairness = 0.10
    w_uncertainty = 0.06
    w_work_density = 0.04
    
    base_score = (
        w_urgency * norm_urgency +
        w_energy * (norm_energy * energy_gate) +
        w_critical * (-norm_progress_velocity) +
        w_fairness * (norm_wait_efficiency * fairness_mask) +
        w_uncertainty * norm_unc_sensitivity +
        w_work_density * (norm_energy_per_mi * work_density_mask)
    )
    
    # Apply hard urgency lockstep
    score = np.where(is_urgent, urgency_score, base_score)
    
    # Final sanitization
    score = np.clip(score, -1000000000000000.0, 1000000000000000.0)
    score = np.nan_to_num(score, nan=1000000000000000.0, posinf=1000000000000000.0, neginf=-1000000000000000.0)
    
    # Assert shape compliance
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
