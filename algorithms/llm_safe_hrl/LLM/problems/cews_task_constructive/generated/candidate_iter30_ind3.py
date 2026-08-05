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
    v2 priority rule: Refined urgency-energy-fairness optimizer with:
      - Hard urgency dominance preserved: all slack<=0 tasks get ultra-low fixed score
      - Restored *slack-normalized* energy density (unclipped) for late-task discrimination
      - Urgency weight increased to 0.49 for stronger deadline fidelity enforcement
      - Energy penalty now scaled by |slack|^{-1} for tighter deadline coupling
      - Starvation rescue refined: uses normalized wait_ratio * (1 + slack/adaptive_slack_thresh)
      - Uncertainty boost enhanced with slack-aware scaling: dur_uncertainty * (1 + max(0,-slack)/tau)
      - Critical work density replaced by *slack-adjusted criticality density* to avoid dilution
      - All robust normalizations retain small-N fallbacks and degeneracy handling
      - Final weights: urgency (0.49), critical latency (0.18), energy (0.17), fairness (0.07),
        starvation (0.05), uncertainty (0.04) — prioritizing hard deadlines and risk-aware energy
    '''
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
            x_min, x_max = (np.min(x_finite), np.max(x_finite))
        else:
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
    duration = min_exec_time + min_comm_time + eps
    risk_adjusted_duration = duration * (1.0 + uncertainty)
    # Restore unclipped slack-normalized energy density for better late-task discrimination
    energy_density = np.divide(min_incremental_energy, risk_adjusted_duration, out=np.zeros_like(min_incremental_energy), where=risk_adjusted_duration != 0)
    energy_density = np.nan_to_num(energy_density, nan=0.0, posinf=0.0, neginf=0.0)
    norm_energy_density = robust_minmax_norm(energy_density)
    is_urgent = (slack <= 0.0).astype(np.float64)
    norm_upward_rank = robust_minmax_norm(upward_rank)
    critical_latency_raw = duration * (1.0 + 0.5 * norm_upward_rank)
    norm_critical_latency = robust_minmax_norm(critical_latency_raw)
    # Adaptive slack threshold for tight-slack masking
    positive_slack = slack[slack > 0]
    adaptive_slack_thresh = np.median(positive_slack) if len(positive_slack) > 0 else 1.0
    tight_slack_mask = (slack <= adaptive_slack_thresh).astype(np.float64)
    high_rank_mask = (upward_rank >= np.median(upward_rank) + eps).astype(np.float64) if N > 1 else np.ones(N)
    # Energy penalty now explicitly scaled by inverse slack magnitude for urgency coupling
    slack_abs_inv = np.divide(1.0, np.abs(slack) + eps, out=np.zeros_like(slack), where=np.abs(slack) + eps != 0)
    slack_abs_inv = np.nan_to_num(slack_abs_inv, nan=0.0, posinf=0.0, neginf=0.0)
    energy_penalty = norm_energy_density * tight_slack_mask * high_rank_mask * slack_abs_inv
    # Fairness penalty: normalized wait-per-duration ratio
    wait_ratio = np.divide(ready_wait_time, duration + eps, out=np.zeros_like(ready_wait_time), where=duration + eps != 0)
    wait_ratio = np.nan_to_num(wait_ratio, nan=0.0, posinf=0.0, neginf=0.0)
    norm_wait_ratio = robust_minmax_norm(wait_ratio)
    ratio_finite = wait_ratio[np.isfinite(wait_ratio)]
    median_ratio = np.median(ratio_finite) if len(ratio_finite) > 0 else 1.0
    wait_gate = (wait_ratio > 1.5 * median_ratio).astype(np.float64)
    wait_penalty = (1.0 - is_urgent) * norm_wait_ratio * wait_gate
    # Starvation rescue: slack-aware wait boost to prioritize long-waiting feasible tasks
    slack_scale = np.where(slack > 0, 1.0 + np.divide(slack, adaptive_slack_thresh + eps, out=np.zeros_like(slack), where=adaptive_slack_thresh + eps != 0), 0.0)
    starvation_boost = np.where((wait_ratio > 1.5 * median_ratio) & (slack > 0.0), wait_ratio * (1.0 + 0.1 * norm_upward_rank) * slack_scale, 0.0)
    norm_starvation = robust_minmax_norm(starvation_boost)
    # Enhanced uncertainty boost: linearly penalizes lateness via dur_uncertainty * (1 + max(0,-slack)/tau)
    dur_uncertainty = np.divide(uncertainty, duration, out=np.zeros_like(uncertainty), where=duration != 0)
    dur_uncertainty = np.nan_to_num(dur_uncertainty, nan=0.0, posinf=0.0, neginf=0.0)
    tau = 15.0
    lateness_bias = 1.0 + np.maximum(0.0, -slack) / tau
    uncertainty_boost = np.clip(dur_uncertainty * lateness_bias, 0.0, 1.0)
    norm_uncertainty_boost = robust_minmax_norm(uncertainty_boost)
    # Slack-adjusted criticality density replaces residual work density
    criticality_density = np.divide(upward_rank, (remaining_work + eps), out=np.zeros_like(upward_rank), where=remaining_work + eps != 0)
    criticality_density = np.nan_to_num(criticality_density, nan=0.0, posinf=0.0, neginf=0.0)
    # Apply slack-aware weighting: higher density for tighter slack
    slack_weight = np.where(slack > 0, 1.0, 2.0)  # Double weight for urgent/late tasks
    slack_adjusted_criticality = criticality_density * slack_weight
    norm_criticality_density = robust_minmax_norm(slack_adjusted_criticality)
    # Weighted combination with increased urgency dominance and energy emphasis
    non_urgent_contrib = 0.18 * norm_critical_latency + 0.17 * energy_penalty + 0.07 * wait_penalty + 0.05 * norm_starvation + 0.04 * norm_uncertainty_boost + 0.04 * norm_criticality_density
    score = np.where(is_urgent, -1000000000000.0, 0.49 + non_urgent_contrib)
    score = np.clip(score, -1000000000000.0, 1000000000000.0)
    score = np.nan_to_num(score, nan=1000000000000.0, posinf=1000000000000.0, neginf=-1000000000000.0)
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
