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
    v2 priority rule: Hard-deadline dominance + adaptive slack-scaled criticality +
                      energy-risk gating with deadline proximity + starvation-resilient fairness +
                      uncertainty-coupling only under dual feasibility + robust normalization +
                      unified lateness dominance + normalized work-consolidation boost.
    
    Key synthesis improvements:
    - Combines Parent 2's adaptive lateness penalty (avoiding overflow, enabling relative urgency)
      with Parent 1's explicit *lateness dominance* via direct assignment (not just additive term).
    - Uses robust minmax normalization (Parent 2) for all components — stable for small N and skewed data.
    - Integrates Parent 1's *uncertainty-normalized work consolidation*: prefers stable VMs for light sub-DAGs,
      implemented as uncertainty / (1 + norm_remaining_work), gated by slack > 0 and criticality.
    - Enhances fairness: uses wait-time scaled by normalized remaining_work *only when slack > median*,
      preventing bias toward long-wait low-importance tasks.
    - Introduces *energy-efficiency boost* under high slack margin (rel_slack > 0.7) to aggressively favor low-energy options.
    - All divisions guarded; NaN/inf replaced deterministically; no in-place mutation; deterministic output.
    """
    eps = 1e-08
    # Safe input conversion and sanitization
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    slack = np.nan_to_num(np.asarray(slack, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=np.float64)
    
    def robust_minmax_norm(x):
        if x.size == 0:
            return np.zeros_like(x)
        if N == 1:
            return np.zeros_like(x)
        p01 = np.percentile(x, 1.0, method='lower')
        p99 = np.percentile(x, 99.0, method='higher')
        x_clipped = np.clip(x, p01, p99)
        x_min = np.min(x_clipped)
        x_max = np.max(x_clipped)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x_clipped - x_min) / (x_max - x_min + eps)
    
    # Lateness dominance: direct high-priority assignment for overdue tasks
    lateness_mask = slack <= 0.0
    abs_slack = np.abs(slack)
    # Adaptive penalty: scales with deficit but avoids numeric explosion
    lateness_penalty = np.where(lateness_mask, -1e12 * (1.0 + 0.05 * abs_slack), 0.0)
    
    # Duration and critical timing
    duration = min_exec_time + min_comm_time + eps
    critical_timing = duration * upward_rank
    norm_critical_timing = robust_minmax_norm(critical_timing)
    
    # Slack scaling: suppress critical bias when slack is tight but positive
    rel_slack = np.divide(slack, duration, out=np.zeros_like(slack), where=duration != 0)
    rel_slack = np.nan_to_num(rel_slack, nan=0.0, posinf=0.0, neginf=0.0)
    slack_scale = np.clip(1.0 - np.clip(rel_slack, 0.0, 1.0), 0.0, 1.0)
    scaled_critical_timing = norm_critical_timing * slack_scale
    
    # Energy density: marginal energy per unit time
    energy_density = np.divide(min_incremental_energy, duration, out=np.zeros_like(min_incremental_energy), where=duration != 0)
    energy_density = np.nan_to_num(energy_density, nan=0.0, posinf=0.0, neginf=0.0)
    norm_energy_density = robust_minmax_norm(energy_density)
    
    # Energy-risk gating: penalize energy density only when deadline proximity increases risk
    slack_proximity = np.clip(0.3 - rel_slack, 0.0, 0.3)
    energy_penalty = norm_energy_density * slack_proximity
    
    # Efficiency boost: actively reward low-energy tasks when slack is ample
    efficiency_boost_mask = (slack > 0.0) & (rel_slack > 0.7)
    efficiency_boost = -0.3 * norm_energy_density * efficiency_boost_mask
    
    # Fairness: boost long-wait tasks only when slack is above median AND they have significant work
    slack_median = np.median(slack) if N > 1 else slack[0]
    rw_median = np.median(remaining_work) if N > 1 else remaining_work[0]
    fairness_mask = (slack > slack_median) & (remaining_work > rw_median)
    norm_wait_time = robust_minmax_norm(ready_wait_time)
    # Scale wait relief by normalized work to avoid starving heavy sub-DAGs
    norm_rw = robust_minmax_norm(remaining_work)
    wait_relief = norm_wait_time / (np.clip(norm_rw, 0.2, 5.0) + eps)
    wait_relief = np.where(np.isfinite(wait_relief), wait_relief, 0.0)
    fairness_boost = wait_relief * fairness_mask
    
    # Uncertainty coupling: only when task is safe (slack > 0), critical (upward_rank > median), and uncertain
    ur_median = np.median(upward_rank) if N > 1 else upward_rank[0]
    unc_median = np.median(uncertainty) if N > 1 else uncertainty[0]
    unc_mask = (slack > 0.0) & (upward_rank > ur_median) & (uncertainty > unc_median)
    norm_uncertainty = robust_minmax_norm(uncertainty)
    unc_coupling = norm_uncertainty * unc_mask
    
    # Uncertainty-normalized work consolidation: prefer stable VMs for light sub-DAGs
    norm_rw_for_unc = np.clip(robust_minmax_norm(remaining_work), 0.0, 10.0)
    unc_work_density = uncertainty / (1.0 + norm_rw_for_unc + eps)
    unc_work_density = np.where(np.isfinite(unc_work_density), unc_work_density, 0.0)
    norm_unc_work_density = robust_minmax_norm(unc_work_density)
    # Gate consolidation boost only under safety & criticality conditions
    unc_work_boost = -0.15 * norm_unc_work_density * unc_mask
    
    # Weighted combination
    score = (
        0.40 * scaled_critical_timing +
        0.20 * robust_minmax_norm(remaining_work) +
        0.15 * energy_penalty +
        0.08 * fairness_boost +
        0.07 * unc_coupling +
        0.05 * unc_work_boost +
        efficiency_boost
    )
    
    # Apply lateness dominance: overwrite score for overdue tasks
    score = np.where(lateness_mask, lateness_penalty, score)
    
    # Final sanitization
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
