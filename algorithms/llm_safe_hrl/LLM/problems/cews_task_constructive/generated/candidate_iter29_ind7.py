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
    v2 priority rule: Hybrid urgency-criticality-energy-fairness with robust risk-gating.
    
    Key innovations:
    - Adaptive hard-deadline enforcement via smooth tanh-based lateness penalty (avoids overflow while preserving dominance)
    - Slack-scaled critical timing: norm_critical_timing * tanh(1.0 - rel_slack) → smoother suppression than step/clamp
    - Energy-efficiency boost gated by positive slack AND high relative slack (>0.5) AND low uncertainty (< median)
    - Fairness rescue uses work-density-normalized wait time, gated only when slack > 0 AND remaining_work > median
    - Uncertainty coupling requires slack > 0 AND upward_rank > median AND uncertainty > median (Parent 2 robustness)
    - All components normalized via 5–95% clipping for stability on small N; all divisions guarded
    - Final score bounded in [-1.0, 1.0] and mapped to [0, 1] for monotonic priority ordering
    """
    eps = 1e-08
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
    
    def robust_5_95_norm(x):
        if x.size == 0:
            return np.zeros_like(x)
        if N == 1:
            return np.zeros_like(x)
        p05 = np.percentile(x, 5.0, method='lower')
        p95 = np.percentile(x, 95.0, method='higher')
        x_clipped = np.clip(x, p05, p95)
        x_min = np.min(x_clipped)
        x_max = np.max(x_clipped)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x_clipped - x_min) / (x_max - x_min + eps)
    
    # Duration and relative slack
    duration = min_exec_time + min_comm_time + eps
    rel_slack = np.divide(slack, duration, out=np.zeros_like(slack), where=duration != 0)
    rel_slack = np.nan_to_num(rel_slack, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Adaptive lateness penalty: smooth tanh(-|slack|/eps) scaled to dominate but avoid overflow
    abs_slack = np.abs(slack)
    lateness_penalty = -np.tanh(abs_slack / (eps + 1.0)) * 1000.0 * (1.0 + 0.05 * abs_slack)
    
    # Critical timing: duration * upward_rank, normalized
    critical_timing = duration * upward_rank
    norm_critical_timing = robust_5_95_norm(critical_timing)
    
    # Slack-scaled critical timing using smooth tanh gate instead of clamp
    slack_gate = np.tanh(np.clip(1.0 - rel_slack, -5.0, 5.0))
    scaled_critical_timing = norm_critical_timing * slack_gate
    
    # Energy density and gating
    energy_density = np.divide(min_incremental_energy, duration, out=np.zeros_like(min_incremental_energy), where=duration != 0)
    energy_density = np.nan_to_num(energy_density, nan=0.0, posinf=0.0, neginf=0.0)
    norm_energy_density = robust_5_95_norm(energy_density)
    
    # Efficiency boost: prefer low-energy tasks only when safe (positive slack, high rel_slack, low uncertainty)
    rw_median = np.median(remaining_work) if N > 1 else remaining_work[0]
    ur_median = np.median(upward_rank) if N > 1 else upward_rank[0]
    unc_median = np.median(uncertainty) if N > 1 else uncertainty[0]
    efficiency_boost_mask = (slack > 0.0) & (rel_slack > 0.5) & (uncertainty < unc_median)
    efficiency_boost = -0.15 * norm_energy_density * efficiency_boost_mask
    
    # Fairness rescue: work-density-normalized wait time, gated by safety and workload
    work_density = np.divide(remaining_work, duration + eps, out=np.zeros_like(remaining_work), where=duration + eps != 0)
    wait_score = np.divide(ready_wait_time, work_density + eps, out=np.zeros_like(ready_wait_time), where=work_density + eps != 0)
    norm_wait_score = robust_5_95_norm(wait_score)
    fairness_mask = (slack > 0.0) & (remaining_work > rw_median)
    fairness_boost = norm_wait_score * fairness_mask
    
    # Uncertainty coupling: only when slack-safe, critical, and uncertain
    unc_mask = (slack > 0.0) & (upward_rank > ur_median) & (uncertainty > unc_median)
    norm_uncertainty = robust_5_95_norm(uncertainty)
    unc_coupling = norm_uncertainty * unc_mask
    
    # Composite score (all components normalized to [0,1] range before weighting)
    score = (
        0.42 * scaled_critical_timing +
        0.23 * robust_5_95_norm(remaining_work) +
        0.14 * norm_energy_density * np.clip(0.3 - rel_slack, 0.0, 0.3) +
        0.09 * fairness_boost +
        0.06 * unc_coupling +
        efficiency_boost
    )
    
    # Apply lateness penalty (dominant term)
    score = score + lateness_penalty
    
    # Final normalization and bounding
    score = np.nan_to_num(score, nan=1.0, posinf=1.0, neginf=-1.0)
    score_min = np.min(score)
    score_max = np.max(score)
    if score_max - score_min < eps:
        final_score = np.zeros_like(score)
    else:
        final_score = (score - score_min) / (score_max - score_min + eps)
    
    # Ensure shape and finiteness
    final_score = np.clip(final_score, 0.0, 1.0)
    assert final_score.shape == (N,), f'Expected shape (N,)={N}, got {final_score.shape}'
    return final_score
