import numpy as np

# 函数名中的 2 由 SeEvo 在生成 Prompt 时替换为目标版本号。
# 八个输入均为长度 N 的一维数组；相同下标始终指向同一个 ready task。
# 返回值也必须是长度 N 的一维数组，并遵守“分数越小，优先级越高”。
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

    '''
    v3 priority rule: Critical-path risk-aware urgency-energy optimizer with:
      - Hard urgency dominance preserved, but augmented with upstream criticality propagation
      - Quantile-aware slack gating (p10/p90 instead of median) for bimodal deadline distributions
      - Critical-path uncertainty aggregation: weighted sum of descendant uncertainties via upward_rank
      - Slack-normalized energy density now uses *critical-path-adjusted* duration (duration * (1 + aggregated_uncertainty))
      - Starvation rescue enhanced with dynamic wait_ratio threshold scaling by slack quantile band
      - Unified normalization using robust quantile clamping (1%-99%) even for N>=2, avoiding median fragility
      - Final weights rebalanced to emphasize critical-path risk (0.52), energy efficiency (0.20), fairness (0.12), starvation (0.10), uncertainty (0.06)
    '''
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

    def robust_quantile_norm(x):
        if x.size == 0:
            return np.zeros_like(x)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x_finite = x[np.isfinite(x)]
        if x_finite.size == 0:
            return np.zeros_like(x)
        # Always use quantile-based clipping (1%–99%) for robustness — avoids median fragility in bimodal cases
        p01 = np.percentile(x_finite, 1.0, method='midpoint')
        p99 = np.percentile(x_finite, 99.0, method='midpoint')
        x_clipped = np.clip(x_finite, p01, p99)
        x_min, x_max = (np.min(x_clipped), np.max(x_clipped))
        if x_max - x_min < eps:
            return np.zeros_like(x)
        normed = np.zeros_like(x)
        valid_mask = np.isfinite(x)
        normed[valid_mask] = (x[valid_mask] - x_min) / (x_max - x_min + eps)
        return normed

    # Compute base duration and critical-path-adjusted uncertainty
    duration = min_exec_time + min_comm_time + eps
    # Aggregate upstream uncertainty: higher upward_rank → more downstream risk → amplify local uncertainty
    # Use normalized upward_rank as weight to avoid bias from absolute scale
    norm_upward_rank = robust_quantile_norm(upward_rank)
    aggregated_uncertainty = uncertainty * (1.0 + 0.3 * norm_upward_rank)
    risk_adjusted_duration = duration * (1.0 + aggregated_uncertainty)

    # Energy density: marginal energy per unit risk-adjusted duration
    energy_density = np.divide(min_incremental_energy, risk_adjusted_duration, 
                               out=np.zeros_like(min_incremental_energy), 
                               where=risk_adjusted_duration != 0)
    energy_density = np.nan_to_num(energy_density, nan=0.0, posinf=0.0, neginf=0.0)

    # Hard urgency dominance: all slack <= 0 get ultra-low score
    is_urgent = (slack <= 0.0).astype(float)

    # Quantile-aware slack gating (robust to bimodal slack distributions)
    slack_finite = slack[np.isfinite(slack)]
    if len(slack_finite) > 0:
        slack_p10 = np.percentile(slack_finite, 10.0, method='midpoint')
        slack_p90 = np.percentile(slack_finite, 90.0, method='midpoint')
        # Tight slack = below p10 (most urgent non-violated); loose = above p90 (very relaxed)
        tight_slack_mask = (slack <= slack_p10).astype(float)
        loose_slack_mask = (slack >= slack_p90).astype(float)
    else:
        tight_slack_mask = np.zeros_like(slack)
        loose_slack_mask = np.zeros_like(slack)

    # Critical latency: duration weighted by critical path importance
    critical_latency_raw = duration * (1.0 + 0.7 * norm_upward_rank)
    norm_critical_latency = robust_quantile_norm(critical_latency_raw)

    # Energy penalty: only penalize high-energy-density tasks that are both critical AND tight-slacked
    norm_energy_density = robust_quantile_norm(energy_density)
    energy_density_finite = energy_density[np.isfinite(energy_density)]
    energy_floor = np.percentile(energy_density_finite, 10) + eps if len(energy_density_finite) > 0 else eps
    energy_significant_mask = (energy_density >= energy_floor).astype(float)
    energy_penalty = norm_energy_density * tight_slack_mask * energy_significant_mask

    # Wait ratio and starvation rescue with slack-band-adaptive threshold
    wait_ratio = np.divide(ready_wait_time, duration + eps, 
                           out=np.zeros_like(ready_wait_time), 
                           where=duration + eps != 0)
    wait_ratio = np.nan_to_num(wait_ratio, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Dynamic threshold: stricter for tight-slack tasks (p10 slack band), looser for loose (p90)
    ratio_finite = wait_ratio[np.isfinite(wait_ratio)]
    median_ratio = np.median(ratio_finite) if len(ratio_finite) > 0 else 1.0
    # Scale threshold by slack quantile position: tighter deadlines → lower wait tolerance
    slack_position = np.interp(slack, 
                               [np.min(slack_finite) if len(slack_finite) > 0 else 0, 
                                np.max(slack_finite) if len(slack_finite) > 0 else 1], 
                               [0.5, 2.0])  # 0.5× for tight, 2.0× for loose
    wait_threshold = median_ratio * np.clip(slack_position, 0.5, 2.0) + eps

    # Critical work density: work per unit criticality — low values indicate "light but critical" tasks prone to starvation
    critical_work_density = np.divide(remaining_work, upward_rank + eps, 
                                      out=np.zeros_like(remaining_work), 
                                      where=upward_rank + eps != 0)
    critical_work_density = np.nan_to_num(critical_work_density, nan=0.0, posinf=0.0, neginf=0.0)
    work_density_finite = critical_work_density[np.isfinite(critical_work_density)]
    median_work_density = np.median(work_density_finite) if len(work_density_finite) > 0 else 0.0

    is_starvable = (wait_ratio > wait_threshold) & (slack > 0.0) & (critical_work_density <= median_work_density + eps)
    starvation_boost = np.where(is_starvable, 
                               wait_ratio * (1.0 + 0.2 * norm_upward_rank), 
                               0.0)
    norm_starvation = robust_quantile_norm(starvation_boost)

    # Uncertainty boost: lateness-aware + criticality-weighted
    dur_uncertainty = np.divide(aggregated_uncertainty, duration + eps, 
                                out=np.zeros_like(aggregated_uncertainty), 
                                where=duration + eps != 0)
    dur_uncertainty = np.nan_to_num(dur_uncertainty, nan=0.0, posinf=0.0, neginf=0.0)
    tau = 10.0  # Faster decay for stronger late-task focus
    lateness_bias = np.exp(-np.maximum(0.0, -slack) / tau)
    uncertainty_boost = np.clip(dur_uncertainty * lateness_bias * (1.0 + 0.4 * norm_upward_rank), 0.0, 1.0)
    norm_uncertainty_boost = robust_quantile_norm(uncertainty_boost)

    # Fairness term: normalized wait time, but suppressed for urgent tasks
    norm_wait_time = robust_quantile_norm(ready_wait_time)
    fairness_term = (1.0 - is_urgent) * norm_wait_time

    # Non-urgent contribution components — weights tuned for critical-path fidelity & energy minimization
    non_urgent_contrib = (
        0.52 * norm_critical_latency +           # dominant: critical path latency risk
        0.20 * energy_penalty +                  # energy efficiency under risk constraints
        0.12 * fairness_term +                   # fairness across ready tasks
        0.10 * norm_starvation +                 # targeted starvation rescue
        0.06 * norm_uncertainty_boost            # uncertainty-aware risk amplification
    )

    # Final score: urgent tasks get highest priority (lowest score); others get weighted composite
    score = np.where(is_urgent, -1e12, 0.0 + non_urgent_contrib)
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)

    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
