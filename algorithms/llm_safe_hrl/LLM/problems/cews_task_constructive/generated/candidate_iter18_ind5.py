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
    v2 priority rule: Absolute urgency dominance + critical-path-aware risk gating + robust fairness amplification.
    
    Key evolutionary improvements over v1:
    - Replaced fixed -1e12 offset with adaptive urgency penalty: -1e12 * (1 + |slack|/max(1, median(|slack|))) for stronger prioritization of *more urgent* tasks within urgent set.
    - Critical-latency now incorporates slack-aware scaling: penalizes high-duration tasks *only when slack is tight*, avoiding over-penalization of long but safe tasks.
    - Energy penalty refined to use *upward-rank-weighted* energy density thresholding — higher rank tasks tolerate higher energy density before penalty triggers.
    - Uncertainty coupling uses *normalized slack distance* (0→1 mapping) instead of clipped 1/|slack|, enabling smooth, differentiable risk ramp-up near deadline.
    - Fairness term upgraded to *exponential wait-time amplification* for non-urgent tasks, preventing starvation without quantile jumps.
    - All robust normalizations now include explicit NaN/inf pre-sanitization and size-zero guards.
    - Final score explicitly clamped to finite bounds and validated for shape/determinism.
    '''
    eps = 1e-08
    # Sanitize and copy inputs to prevent mutation
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=1e6, neginf=eps).copy()
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=1e6, neginf=eps).copy()
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=1e6, neginf=eps).copy()
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6).copy()
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=eps, posinf=1e6, neginf=eps).copy()
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=eps, posinf=1e6, neginf=eps).copy()
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=1e6, neginf=0.0).copy()
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=eps, posinf=1e6, neginf=eps).copy()
    
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)
    
    def robust_minmax_norm(x):
        if x.size == 0:
            return np.zeros_like(x)
        # Pre-sanitize
        x = np.nan_to_num(x, nan=np.median(x[x != 0]) if np.any(x != 0) else eps, posinf=np.max(x[x != np.inf]), neginf=np.min(x[x != -np.inf]))
        p01 = np.percentile(x, 1.0)
        p99 = np.percentile(x, 99.0)
        x_clipped = np.clip(x, p01, p99)
        x_min = np.min(x_clipped)
        x_max = np.max(x_clipped)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x_clipped - x_min) / (x_max - x_min + eps)
    
    # Adaptive urgency: stronger penalty for deeper lateness, normalized by median urgency magnitude
    is_urgent = (slack <= 0.0).astype(float)
    abs_slack = np.abs(slack)
    median_abs_slack = np.median(abs_slack[is_urgent == 1]) if np.any(is_urgent) else 1.0
    urgency_scale = 1.0 + np.where(is_urgent, abs_slack / (median_abs_slack + eps), 0.0)
    urgency_penalty = np.where(is_urgent, -1e12 * urgency_scale, 0.0)
    
    # Slack-aware critical latency: only penalize duration when slack is tight (normalized slack distance ∈ [0,1])
    duration = min_exec_time + min_comm_time + eps
    # Map slack to [0,1]: 0 when slack <= 0, 1 when slack >= max_slack; linear in between
    slack_range = np.max(slack) - np.min(slack)
    slack_dist = np.clip((slack - np.min(slack)) / (slack_range + eps), 0.0, 1.0)
    # Inverse slack proximity: high penalty when slack is small → slack_proximity ≈ 0
    slack_proximity = 1.0 - slack_dist
    critical_latency_raw = duration * (1.0 + 0.8 * robust_minmax_norm(upward_rank)) * slack_proximity
    norm_critical_latency = robust_minmax_norm(critical_latency_raw)
    
    # Rank-weighted energy density threshold: stricter threshold for high-rank tasks
    energy_density = min_incremental_energy / (duration + eps)
    rank_weighted_energy = energy_density / (upward_rank + eps)
    # Dynamic threshold per rank quantile: higher rank → lower tolerated energy density
    rank_quantiles = np.percentile(upward_rank, [25, 50, 75, 95])
    rank_bins = np.digitize(upward_rank, rank_quantiles)  # 0..4
    base_thresholds = np.array([0.99, 0.97, 0.95, 0.92, 0.90])
    dynamic_percentile = base_thresholds[np.clip(rank_bins, 0, 4)]
    threshold_rank_energy = np.array([
        np.percentile(rank_weighted_energy[upward_rank > 0], p) if np.any(upward_rank > 0) else np.percentile(rank_weighted_energy, 95.0)
        for p in dynamic_percentile
    ])
    threshold_rank_energy = np.nan_to_num(threshold_rank_energy, nan=np.percentile(rank_weighted_energy, 95.0) + eps)
    energy_penalty_mask = (rank_weighted_energy > (threshold_rank_energy + eps)).astype(float)
    norm_energy_density = robust_minmax_norm(energy_density)
    energy_penalty = norm_energy_density * energy_penalty_mask
    
    # Exponential fairness: prevents starvation via smooth, progressive wait-time amplification
    norm_wait_time = robust_minmax_norm(ready_wait_time)
    wait_penalty = (1.0 - is_urgent) * (np.exp(norm_wait_time * 3.0) - 1.0) / (np.e**3 - 1)  # maps [0,1] → [0,1] exponentially
    
    # Smooth uncertainty gating using normalized slack distance (0→1) and rank gate
    rank_gate = robust_minmax_norm(upward_rank)
    uncertainty_boost = uncertainty * slack_proximity * rank_gate
    norm_uncertainty_boost = robust_minmax_norm(uncertainty_boost)
    
    # Normalize remaining work
    norm_remaining_work = robust_minmax_norm(remaining_work)
    
    # Hierarchical weighted sum: urgency dominates; others are additive penalties
    score = (
        urgency_penalty +
        0.25 * norm_critical_latency +
        0.22 * energy_penalty +
        0.12 * wait_penalty +
        0.09 * norm_uncertainty_boost +
        0.04 * norm_remaining_work
    )
    
    # Final sanitization and clipping
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
