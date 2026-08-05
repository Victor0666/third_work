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
    v2 priority rule: Hard urgency dominance + critical-path energy gating + 
    adaptive fairness + uncertainty-aware slack normalization + robust outlier resilience.
    
    Key improvements over parents:
    - Combines Parent 2's guaranteed urgent-task priority (-1e12) and robust min-max norm
      with Parent 1's MAD-based robust_zscore for wait_efficiency & progress_velocity (more resilient to skew).
    - Introduces *dynamic urgency threshold* using 10th percentile of slack instead of binary (slack <= 0),
      enabling proactive scheduling before hard deadline violation while preserving hard constraint semantics.
    - Replaces fixed energy penalty mask with *graded criticality coupling*: energy_penalty = 
      norm_energy_density * sigmoid(0.5 * (upward_rank / median_upward_rank) + 2.0 * (1 - norm_rel_slack)).
    - Uncertainty coupling uses relative slack *and* upward rank *and* normalized wait time to suppress noise
      on low-urgency/low-criticality tasks, avoiding spurious penalties.
    - All normalizations fallback-safe: degenerate ranges → zero; NaN/inf → deterministic replacement.
    - Final weights sum to 1.0: urgency (0.38) > critical_path_energy (0.25) > wait_fairness (0.14) > 
      uncertainty_slack (0.10) > progress_velocity (0.08) > work_bias (0.05).
    """
    eps = 1e-08
    # Ensure float64 for numerical stability and copy to avoid in-place modification
    min_exec_time = np.asarray(min_exec_time, dtype=np.float64).copy()
    min_comm_time = np.asarray(min_comm_time, dtype=np.float64).copy()
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=np.float64).copy()
    slack = np.asarray(slack, dtype=np.float64).copy()
    upward_rank = np.asarray(upward_rank, dtype=np.float64).copy()
    remaining_work = np.asarray(remaining_work, dtype=np.float64).copy()
    ready_wait_time = np.asarray(ready_wait_time, dtype=np.float64).copy()
    uncertainty = np.asarray(uncertainty, dtype=np.float64).copy()
    
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=np.float64)
    
    # Robust min-max normalization with 1%-99% clipping
    def robust_minmax_norm(x):
        if x.size == 0:
            return np.zeros_like(x)
        p01 = np.percentile(x, 1.0, method='midpoint')
        p99 = np.percentile(x, 99.0, method='midpoint')
        x_clipped = np.clip(x, p01, p99)
        x_min = np.min(x_clipped)
        x_max = np.max(x_clipped)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x_clipped - x_min) / (x_max - x_min + eps)
    
    # MAD-based robust z-score for wait_efficiency and progress_velocity (Parent 1 strength)
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
    
    # Duration and derived metrics
    duration = min_exec_time + min_comm_time + eps
    rel_slack = np.divide(slack, duration, out=np.zeros_like(slack), where=duration != 0)
    rel_slack = np.where(np.isfinite(rel_slack), rel_slack, 0.0)
    
    # Dynamic urgency: proactive threshold at 10th percentile slack (not just slack <= 0)
    if N > 1:
        slack_10p = np.percentile(slack, 10.0, method='midpoint')
        is_urgent = (slack <= slack_10p).astype(np.float64)
    else:
        is_urgent = np.ones(N, dtype=np.float64)
    
    # Urgency score: guaranteed minimum for urgent tasks
    urgency_score = np.full(N, -1e12, dtype=np.float64)
    
    # Critical path latency (Parent 2)
    critical_latency_raw = duration * (1.0 + 0.7 * robust_minmax_norm(upward_rank))
    norm_critical_latency = robust_minmax_norm(critical_latency_raw)
    
    # Energy density and graded criticality-gated penalty (novel: sigmoid coupling)
    energy_density = np.divide(min_incremental_energy, duration + eps, out=np.zeros_like(min_incremental_energy), where=duration + eps != 0)
    energy_density = np.where(np.isfinite(energy_density), energy_density, 0.0)
    norm_energy_density = robust_minmax_norm(energy_density)
    
    # Graded energy penalty: increases with upward_rank and decreases with rel_slack (proactive)
    median_upward = np.median(upward_rank) + eps
    norm_upward_rank = robust_minmax_norm(upward_rank)
    norm_rel_slack = robust_minmax_norm(np.clip(rel_slack, -10.0, 10.0))  # prevent extreme values
    # Sigmoid gate: high when rank is high AND slack is low (i.e., tight critical path)
    energy_gate = 1.0 / (1.0 + np.exp(-0.5 * (upward_rank / median_upward) - 2.0 * (1.0 - norm_rel_slack)))
    energy_penalty = norm_energy_density * energy_gate
    
    # Wait fairness: robust z-score for efficiency + dynamic work-threshold (hybrid)
    wait_efficiency = np.divide(ready_wait_time, remaining_work + eps, out=np.zeros_like(ready_wait_time), where=remaining_work + eps != 0)
    wait_efficiency = np.where(np.isfinite(wait_efficiency), wait_efficiency, 0.0)
    norm_wait_efficiency = robust_zscore(wait_efficiency)
    
    # Work threshold for fairness gating (Parent 2)
    work_threshold = np.percentile(remaining_work, 5.0, method='midpoint') + eps
    wait_gate = (remaining_work >= work_threshold).astype(np.float64)
    wait_penalty = (1.0 - is_urgent) * norm_wait_efficiency * wait_gate
    
    # Uncertainty coupling: only activates under urgency & criticality (suppresses noise)
    abs_rel_slack = np.abs(rel_slack) + eps
    slack_sensitivity = np.clip(1.0 / abs_rel_slack, 0.1, 10.0)
    rank_sensitivity = robust_minmax_norm(upward_rank)
    wait_sensitivity = robust_minmax_norm(ready_wait_time)  # normalize wait time magnitude
    uncertainty_boost = uncertainty * slack_sensitivity * rank_sensitivity * wait_sensitivity
    norm_uncertainty_boost = robust_minmax_norm(uncertainty_boost)
    
    # Progress velocity: HEFT-style importance per unit time (Parent 1)
    progress_velocity = np.divide(upward_rank, duration, out=np.zeros_like(upward_rank), where=duration != 0)
    norm_progress_velocity = robust_zscore(progress_velocity)
    
    # Remaining work bias (small weight to prefer larger sub-DAGs early)
    norm_remaining_work = robust_minmax_norm(remaining_work)
    
    # Base score components with calibrated weights (sum to 1.0)
    base_score = (
        0.38 * (1.0 - robust_minmax_norm(np.clip(-slack, 0.0, np.inf))) +  # urgency: higher -slack → lower score
        0.25 * energy_penalty +
        0.14 * wait_penalty +
        0.10 * norm_uncertainty_boost +
        0.08 * (-norm_progress_velocity) +  # higher velocity → lower score
        0.05 * norm_remaining_work
    )
    
    # Apply urgent-task override
    score = np.where(is_urgent, urgency_score, base_score)
    
    # Final sanitization
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    
    # Assert shape compliance
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
