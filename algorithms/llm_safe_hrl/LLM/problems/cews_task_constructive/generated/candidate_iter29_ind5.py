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
    v2 priority rule: Hard-deadline-respecting urgency-energy-criticality fusion with monotonic risk coupling.
    
    Key design principles:
    - Strict hard-urgency dominance: all slack <= 0 tasks get ultra-low score (-1e12)
    - Slack-adaptive thresholds (median-based) for robustness to workload skew
    - Risk-aware uncertainty coupling: exp(-max(0,-slack)/tau) for graded late-task penalty
    - Criticality-energy coupling via slack-normalized energy density
    - Starvation rescue gated by wait_ratio > 1.5*median AND positive slack AND low critical work density
    - Unified energy density normalization using risk-adjusted duration (duration * (1+uncertainty))
    - All operations guarded against zero/Nan/inf; deterministic and side-effect free
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
    
    def robust_minmax_norm(x):
        if x.size == 0:
            return np.zeros_like(x)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x_finite = x[np.isfinite(x)]
        if x_finite.size == 0:
            return np.zeros_like(x)
        if x_finite.size < 3:
            x_min, x_max = np.min(x_finite), np.max(x_finite)
        else:
            p01 = np.percentile(x_finite, 1.0, method='midpoint')
            p99 = np.percentile(x_finite, 99.0, method='midpoint')
            x_clipped = np.clip(x_finite, p01, p99)
            x_min, x_max = np.min(x_clipped), np.max(x_clipped)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        normed = np.zeros_like(x)
        valid_mask = np.isfinite(x)
        normed[valid_mask] = (x[valid_mask] - x_min) / (x_max - x_min + eps)
        return normed
    
    # Compute risk-adjusted duration: base duration amplified by uncertainty
    duration = min_exec_time + min_comm_time + eps
    risk_adjusted_duration = duration * (1.0 + uncertainty)
    
    # Energy density: marginal energy per risk-adjusted time unit
    energy_density = np.divide(
        min_incremental_energy,
        risk_adjusted_duration,
        out=np.zeros_like(min_incremental_energy),
        where=risk_adjusted_duration != 0
    )
    energy_density = np.nan_to_num(energy_density, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Urgency flag: strict hard constraint enforcement
    is_urgent = (slack <= 0.0).astype(float)
    
    # Adaptive slack threshold for tightness gating (median of positive slack)
    positive_slack = slack[slack > 0]
    adaptive_slack_thresh = np.median(positive_slack) if len(positive_slack) > 0 else 1.0
    tight_slack_mask = (slack <= adaptive_slack_thresh).astype(float)
    
    # Normalized criticality metrics
    norm_upward_rank = robust_minmax_norm(upward_rank)
    critical_latency_raw = duration * (1.0 + 0.5 * norm_upward_rank)
    norm_critical_latency = robust_minmax_norm(critical_latency_raw)
    
    # Normalize energy density
    norm_energy_density = robust_minmax_norm(energy_density)
    
    # Energy penalty: only applied to high-criticality, tight-slack, significant-energy tasks
    energy_density_finite = energy_density[np.isfinite(energy_density)]
    energy_floor = np.percentile(energy_density_finite, 10) + eps if len(energy_density_finite) > 0 else eps
    energy_significant_mask = (energy_density >= energy_floor).astype(float)
    high_rank_mask = (upward_rank >= np.median(upward_rank) + eps).astype(float) if N > 1 else np.ones(N)
    energy_penalty = norm_energy_density * tight_slack_mask * high_rank_mask * energy_significant_mask
    
    # Starvation detection: wait_ratio > 1.5*median AND positive slack AND low critical work density
    wait_ratio = np.divide(
        ready_wait_time,
        duration + eps,
        out=np.zeros_like(ready_wait_time),
        where=duration + eps != 0
    )
    wait_ratio = np.nan_to_num(wait_ratio, nan=0.0, posinf=0.0, neginf=0.0)
    ratio_finite = wait_ratio[np.isfinite(wait_ratio)]
    median_ratio = np.median(ratio_finite) if len(ratio_finite) > 0 else 1.0
    critical_work_density = np.divide(
        remaining_work,
        upward_rank + eps,
        out=np.zeros_like(remaining_work),
        where=upward_rank + eps != 0
    )
    critical_work_density = np.nan_to_num(critical_work_density, nan=0.0, posinf=0.0, neginf=0.0)
    work_density_finite = critical_work_density[np.isfinite(critical_work_density)]
    median_work_density = np.median(work_density_finite) if len(work_density_finite) > 0 else 0.0
    is_starvable = (
        (wait_ratio > 1.5 * median_ratio) & 
        (slack > 0.0) & 
        (critical_work_density <= median_work_density + eps)
    )
    starvation_boost = np.where(is_starvable, wait_ratio * (1.0 + 0.1 * norm_upward_rank), 0.0)
    norm_starvation = robust_minmax_norm(starvation_boost)
    
    # Uncertainty boost: graded penalty for lateness severity
    dur_uncertainty = np.divide(
        uncertainty,
        duration,
        out=np.zeros_like(uncertainty),
        where=duration != 0
    )
    dur_uncertainty = np.nan_to_num(dur_uncertainty, nan=0.0, posinf=0.0, neginf=0.0)
    tau = 15.0
    lateness_bias = np.exp(-np.maximum(0.0, -slack) / tau)
    uncertainty_boost = np.clip(dur_uncertainty * lateness_bias, 0.0, 1.0)
    norm_uncertainty_boost = robust_minmax_norm(uncertainty_boost)
    
    # Non-urgent contribution components with rebalanced weights
    non_urgent_contrib = (
        0.21 * norm_critical_latency +
        0.16 * energy_penalty +
        0.08 * (1.0 - is_urgent) * robust_minmax_norm(ready_wait_time) +
        0.05 * norm_starvation +
        0.03 * norm_uncertainty_boost +
        0.03 * robust_minmax_norm(critical_work_density)
    )
    
    # Final score: ultra-low for urgent tasks, otherwise weighted sum
    score = np.where(is_urgent, -1000000000000.0, 0.47 + non_urgent_contrib)
    
    # Clip and sanitize
    score = np.clip(score, -1000000000000.0, 1000000000000.0)
    score = np.nan_to_num(score, nan=1000000000000.0, posinf=1000000000000.0, neginf=-1000000000000.0)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
