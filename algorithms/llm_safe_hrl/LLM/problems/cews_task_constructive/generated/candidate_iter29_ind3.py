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
    v2 priority rule: Hybrid urgency-energy-fairness optimizer with:
      - Hard urgency dominance: all slack<=0 tasks receive fixed ultra-low score (no gating)
      - Slack-normalized criticality-energy coupling via risk-adjusted duration
      - Starvation rescue gated by relative wait, positive slack, and work density
      - Uncertainty boost using exp(-max(0,-slack)/tau) for late-task risk awareness
      - Feasibility-aware energy density clipping at 99th percentile to suppress outliers
      - Robust normalization with explicit small-N fallbacks and degeneracy handling
      - Final weights prioritizing deadline fidelity (0.48), energy efficiency (0.20),
        critical latency (0.17), fairness (0.07), starvation (0.05), uncertainty (0.03)
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

    # Compute base metrics
    duration = min_exec_time + min_comm_time + eps
    risk_adjusted_duration = duration * (1.0 + uncertainty)
    
    # Feasibility-aware energy density with outlier clipping
    energy_density = np.divide(min_incremental_energy, risk_adjusted_duration, 
                              out=np.zeros_like(min_incremental_energy), 
                              where=risk_adjusted_duration != 0)
    energy_density = np.nan_to_num(energy_density, nan=0.0, posinf=0.0, neginf=0.0)
    energy_density_finite = energy_density[np.isfinite(energy_density)]
    if len(energy_density_finite) > 0:
        energy_p99 = np.percentile(energy_density_finite, 99.0, method='midpoint') + eps
        energy_density_clipped = np.clip(energy_density, 0.0, energy_p99)
    else:
        energy_density_clipped = energy_density
    norm_energy_density = robust_minmax_norm(energy_density_clipped)

    # Urgency handling: hard dominance for overdue tasks
    is_urgent = (slack <= 0.0).astype(np.float64)
    
    # Critical latency: duration weighted by rank importance
    norm_upward_rank = robust_minmax_norm(upward_rank)
    critical_latency_raw = duration * (1.0 + 0.5 * norm_upward_rank)
    norm_critical_latency = robust_minmax_norm(critical_latency_raw)

    # Energy penalty only for tasks under tight slack or high criticality
    positive_slack = slack[slack > 0]
    adaptive_slack_thresh = np.median(positive_slack) if len(positive_slack) > 0 else 1.0
    tight_slack_mask = (slack <= adaptive_slack_thresh).astype(np.float64)
    high_rank_mask = (upward_rank >= np.median(upward_rank) + eps).astype(np.float64) if N > 1 else np.ones(N)
    energy_floor = np.percentile(energy_density_finite, 10) + eps if len(energy_density_finite) > 0 else eps
    energy_significant_mask = (energy_density >= energy_floor).astype(np.float64)
    energy_penalty = norm_energy_density * tight_slack_mask * high_rank_mask * energy_significant_mask

    # Fairness: wait-based starvation rescue
    wait_per_work = np.divide(ready_wait_time, remaining_work + eps, 
                             out=np.zeros_like(ready_wait_time), 
                             where=remaining_work + eps != 0)
    wait_per_work = np.nan_to_num(wait_per_work, nan=0.0, posinf=0.0, neginf=0.0)
    norm_wait_per_work = robust_minmax_norm(wait_per_work)
    wpw_finite = wait_per_work[np.isfinite(wait_per_work)]
    wait_threshold = np.median(wpw_finite) * 1.5 + eps if len(wpw_finite) > 0 else eps
    wait_gate = (wait_per_work >= wait_threshold).astype(np.float64)
    wait_penalty = (1.0 - is_urgent) * norm_wait_per_work * wait_gate

    # Starvation boost: only for non-urgent, high-wait, low-work-density tasks
    wait_ratio = np.divide(ready_wait_time, duration + eps, 
                          out=np.zeros_like(ready_wait_time), 
                          where=duration + eps != 0)
    wait_ratio = np.nan_to_num(wait_ratio, nan=0.0, posinf=0.0, neginf=0.0)
    ratio_finite = wait_ratio[np.isfinite(wait_ratio)]
    median_ratio = np.median(ratio_finite) if len(ratio_finite) > 0 else 1.0
    work_density = np.divide(remaining_work, upward_rank + eps, 
                            out=np.zeros_like(remaining_work), 
                            where=upward_rank + eps != 0)
    work_density = np.nan_to_num(work_density, nan=0.0, posinf=0.0, neginf=0.0)
    median_work_density = np.median(work_density[np.isfinite(work_density)]) if np.any(np.isfinite(work_density)) else 0.0
    is_starvable = (wait_ratio > 1.5 * median_ratio) & (slack > 0.0) & (work_density <= median_work_density + eps)
    starvation_boost = np.where(is_starvable, wait_ratio * (1.0 + 0.1 * norm_upward_rank), 0.0)
    norm_starvation = robust_minmax_norm(starvation_boost)

    # Uncertainty boost with late-task awareness
    dur_uncertainty = np.divide(uncertainty, duration, 
                               out=np.zeros_like(uncertainty), 
                               where=duration != 0)
    dur_uncertainty = np.nan_to_num(dur_uncertainty, nan=0.0, posinf=0.0, neginf=0.0)
    tau = 15.0
    lateness_bias = np.exp(-np.maximum(0.0, -slack) / tau)
    uncertainty_boost = np.clip(dur_uncertainty * lateness_bias, 0.0, 1.0)
    norm_uncertainty_boost = robust_minmax_norm(uncertainty_boost)

    # Critical work density: remaining work normalized by criticality
    critical_work_density = np.divide(remaining_work, upward_rank + eps, 
                                     out=np.zeros_like(remaining_work), 
                                     where=upward_rank + eps != 0)
    critical_work_density = np.nan_to_num(critical_work_density, nan=0.0, posinf=0.0, neginf=0.0)
    norm_critical_work_density = robust_minmax_norm(critical_work_density)

    # Combine components with rebalanced weights
    non_urgent_contrib = (
        0.17 * norm_critical_latency +
        0.20 * energy_penalty +
        0.07 * wait_penalty +
        0.05 * norm_starvation +
        0.03 * norm_uncertainty_boost +
        0.03 * norm_critical_work_density
    )
    
    # Base score for non-urgent tasks; ultra-low for urgent ones
    score = np.where(is_urgent, -1000000000000.0, 0.48 + non_urgent_contrib)
    
    # Clip and sanitize
    score = np.clip(score, -1000000000000.0, 1000000000000.0)
    score = np.nan_to_num(score, nan=1000000000000.0, posinf=1000000000000.0, neginf=-1000000000000.0)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
