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
    v2 priority rule: Hybrid urgency-energy-criticality with adaptive banding,
    risk-weighted fairness, and outlier-resilient normalization.
    
    Key improvements:
    - Combines Parent 2's stable robust_minmax_norm (5–95% clipping) with Parent 1's
      dynamic urgency bands (urgent/critical/relaxed) for graded deadline pressure.
    - Uses risk-adjusted energy density: min_incremental_energy / (duration * (1 + uncertainty)^2)
      to strongly penalize high-uncertainty low-energy options near deadlines.
    - Introduces criticality-gated wait relief: only penalizes starvation when slack > median_slack
      AND upward_rank is high, preventing unfairness under tight deadlines.
    - Replaces static work_density with normalized "critical-path workload density":
      (remaining_work * upward_rank) / (duration + eps), biasing consolidation of heavy critical sub-DAGs.
    - All masks and weights are deterministic, finite, and shape-preserving; no in-place ops.
    - Strict sanitization: np.nan_to_num with explicit fill values, eps-guarded divisions, and clipping.
    """
    eps = 1e-08
    # Ensure float64 for numerical stability and cast inputs safely
    min_exec_time = np.asarray(min_exec_time, dtype=np.float64)
    min_comm_time = np.asarray(min_comm_time, dtype=np.float64)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slack = np.asarray(slack, dtype=np.float64)
    upward_rank = np.asarray(upward_rank, dtype=np.float64)
    remaining_work = np.asarray(remaining_work, dtype=np.float64)
    ready_wait_time = np.asarray(ready_wait_time, dtype=np.float64)
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=np.float64)
    
    # Sanitize all inputs: replace NaN/inf with safe defaults
    inputs = [min_exec_time, min_comm_time, min_incremental_energy, slack, 
              upward_rank, remaining_work, ready_wait_time, uncertainty]
    cleaned = [np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0) for arr in inputs]
    min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, \
        remaining_work, ready_wait_time, uncertainty = cleaned
    
    # Robust min-max normalization (5%-95% clipping) for small-N stability
    def robust_minmax_norm(x):
        if x.size == 0:
            return np.zeros_like(x)
        p05 = np.percentile(x, 5.0)
        p95 = np.percentile(x, 95.0)
        x_clipped = np.clip(x, p05, p95)
        x_min = np.min(x_clipped)
        x_max = np.max(x_clipped)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x_clipped - x_min) / (x_max - x_min + eps)
    
    # Dynamic urgency bands based on slack distribution
    slack_clean = np.nan_to_num(slack, nan=0.0, posinf=0.0, neginf=0.0)
    median_slack = np.median(slack_clean) if N > 0 else 0.0
    is_urgent = (slack_clean <= 0.0).astype(np.float64)
    is_critical = ((slack_clean > 0.0) & (slack_clean <= median_slack)).astype(np.float64)
    is_relaxed = (slack_clean > median_slack).astype(np.float64)
    
    # Duration and risk-adjusted energy density
    duration = min_exec_time + min_comm_time + eps
    effective_duration = duration * (1.0 + uncertainty + eps) ** 2
    risk_adj_energy_density = np.divide(
        min_incremental_energy, 
        effective_duration, 
        out=np.zeros_like(min_incremental_energy), 
        where=effective_duration != 0
    )
    risk_adj_energy_density = np.where(np.isfinite(risk_adj_energy_density), risk_adj_energy_density, 0.0)
    
    # Critical-path workload density: favors heavy & critical tasks on stable VMs
    cp_work_density = (remaining_work * (upward_rank + eps)) / (duration + eps)
    cp_work_density = np.where(np.isfinite(cp_work_density), cp_work_density, 0.0)
    
    # Criticality-gated wait relief: only applies in relaxed band and for high-rank tasks
    q75_rank = np.percentile(upward_rank, 75) if N > 0 else 0.0
    high_rank_mask = (upward_rank > q75_rank).astype(np.float64)
    wait_relief = np.divide(
        ready_wait_time, 
        (remaining_work + eps), 
        out=np.zeros_like(ready_wait_time), 
        where=remaining_work + eps != 0
    )
    wait_relief = np.where(np.isfinite(wait_relief), wait_relief, 0.0)
    gated_wait_relief = wait_relief * is_relaxed * high_rank_mask * (1.0 + uncertainty)
    
    # Normalize components
    norm_energy = robust_minmax_norm(risk_adj_energy_density)
    norm_cp_work = robust_minmax_norm(cp_work_density)
    norm_wait = robust_minmax_norm(gated_wait_relief)
    
    # Base score: minimize energy density in critical/relaxed bands; highest priority for urgent
    base_score = (
        0.42 * norm_energy * (is_critical + is_relaxed) + 
        0.28 * norm_cp_work + 
        0.15 * norm_wait * is_relaxed + 
        0.15 * (1.0 - robust_minmax_norm(upward_rank)) * is_critical
    )
    
    # Urgent tasks get top priority (lowest score)
    score = np.where(is_urgent, -1e15, base_score)
    
    # Clip and sanitize final score
    score = np.clip(score, -1e15, 1e15)
    score = np.nan_to_num(score, nan=1e15, posinf=1e15, neginf=-1e15)
    
    # Final shape assertion
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
