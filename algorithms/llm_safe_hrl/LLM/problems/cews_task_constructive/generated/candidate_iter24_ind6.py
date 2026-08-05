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

    """
    v3 priority rule: Robust urgency-first + adaptive critical-path gating + 
                      slack-aware starvation rescue + degenerate-safe normalization.
    
    Key improvements over v1:
    - Replaces percentile-based q25/q50 gating with *adaptive slack threshold*: 
      uses max(eps, median(slack[slack > 0])) for tightness when slack has positives, else 1.0s.
    - Starvation rescue now gated by *both* slack > 0 AND relative wait ratio > 1.5 * median_ratio,
      preventing late-task bias; uses median instead of percentile for N-robustness.
    - All normalizations use *minmax with explicit fallback to zeros* on degeneracy (no percentile when N<3).
    - Critical latency penalty scaled by (1 + 0.5 * normalized_upward_rank) instead of fixed 0.7 → tighter coupling.
    - Uncertainty boost weighted by *normalized slack proximity* and *dur_uncertainty*, but capped at 1.0.
    - Energy penalty mask now requires *both* tight slack AND high rank AND non-negligible energy density (> p10).
    - Final weights rebalanced: urgency (0.48), critical latency (0.22), gated energy (0.14), 
      fairness (0.09), starvation (0.04), uncertainty (0.03).
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
        # Degenerate-safe: avoid percentile for small N (< 3) → use min/max directly
        if x_finite.size < 3:
            x_min, x_max = np.min(x_finite), np.max(x_finite)
        else:
            p01 = np.percentile(x_finite, 1.0, method='midpoint')
            p99 = np.percentile(x_finite, 99.0, method='midpoint')
            x_clipped = np.clip(x_finite, p01, p99)
            x_min, x_max = np.min(x_clipped), np.max(x_clipped)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        # Broadcast norm to original shape using finite mask
        normed = np.zeros_like(x)
        valid_mask = np.isfinite(x)
        normed[valid_mask] = (x[valid_mask] - x_min) / (x_max - x_min + eps)
        return normed
    
    duration = min_exec_time + min_comm_time + eps
    dur_uncertainty = np.divide(uncertainty, duration, out=np.zeros_like(uncertainty), where=duration != 0)
    dur_uncertainty = np.nan_to_num(dur_uncertainty, nan=0.0, posinf=0.0, neginf=0.0)
    
    is_urgent = (slack <= 0.0).astype(float)
    
    # Adaptive slack threshold: median of positive slack, fallback to 1.0s for all-negative or singleton
    positive_slack = slack[slack > 0]
    if len(positive_slack) > 0:
        adaptive_slack_thresh = np.median(positive_slack)
    else:
        adaptive_slack_thresh = 1.0
    tight_slack_mask = (slack <= adaptive_slack_thresh).astype(float)
    
    # Critical latency: stronger upward-rank coupling, robust norm
    norm_upward_rank = robust_minmax_norm(upward_rank)
    critical_latency_raw = duration * (1.0 + 0.5 * norm_upward_rank)
    norm_critical_latency = robust_minmax_norm(critical_latency_raw)
    
    # Energy density & adaptive gating
    energy_density = np.divide(min_incremental_energy, duration + eps, 
                              out=np.zeros_like(min_incremental_energy), 
                              where=duration + eps != 0)
    norm_energy_density = robust_minmax_norm(energy_density)
    # Energy penalty only if high rank, tight slack, *and* energy density above noise floor
    energy_density_finite = energy_density[np.isfinite(energy_density)]
    energy_floor = np.percentile(energy_density_finite, 10) + eps if len(energy_density_finite) > 0 else eps
    energy_significant_mask = (energy_density >= energy_floor).astype(float)
    high_rank_mask = (upward_rank >= np.median(upward_rank) + eps).astype(float) if N > 1 else np.ones(N)
    energy_penalty_mask = tight_slack_mask * high_rank_mask * energy_significant_mask
    energy_penalty = norm_energy_density * energy_penalty_mask
    
    # Fairness: normalized wait-per-work with median-threshold gating
    wait_per_work = np.divide(ready_wait_time, remaining_work + eps, 
                             out=np.zeros_like(ready_wait_time), 
                             where=remaining_work + eps != 0)
    wait_per_work = np.nan_to_num(wait_per_work, nan=0.0, posinf=0.0, neginf=0.0)
    norm_wait_per_work = robust_minmax_norm(wait_per_work)
    wpw_finite = wait_per_work[np.isfinite(wait_per_work)]
    wait_threshold = np.median(wpw_finite) * 1.5 + eps if len(wpw_finite) > 0 else eps
    wait_gate = (wait_per_work >= wait_threshold).astype(float)
    wait_penalty = (1.0 - is_urgent) * norm_wait_per_work * wait_gate
    
    # Starvation rescue: only for non-urgent, well-ahead-of-deadline tasks with excessive wait
    wait_ratio = np.divide(ready_wait_time, duration + eps, 
                          out=np.zeros_like(ready_wait_time), 
                          where=duration + eps != 0)
    wait_ratio = np.nan_to_num(wait_ratio, nan=0.0, posinf=0.0, neginf=0.0)
    ratio_finite = wait_ratio[np.isfinite(wait_ratio)]
    median_ratio = np.median(ratio_finite) if len(ratio_finite) > 0 else 1.0
    is_starvable = (wait_ratio > 1.5 * median_ratio) & (slack > 0.0) & (upward_rank >= np.median(upward_rank))
    starvation_boost = np.where(is_starvable, wait_ratio * (1.0 + 0.1 * norm_upward_rank), 0.0)
    norm_starvation = robust_minmax_norm(starvation_boost)
    
    # Uncertainty boost: proximity-weighted and capped
    tau = 15.0
    proximity_bias = np.exp(-np.maximum(0.0, slack) / tau)
    uncertainty_boost = np.clip(dur_uncertainty * proximity_bias, 0.0, 1.0)
    norm_uncertainty_boost = robust_minmax_norm(uncertainty_boost)
    
    # Base score and composition
    base_score = np.full(N, 1.0, dtype=float)
    score = np.where(is_urgent, -1000000000000.0, base_score)
    
    non_urgent_contrib = (
        0.22 * norm_critical_latency +
        0.14 * energy_penalty +
        0.09 * wait_penalty +
        0.04 * norm_starvation +
        0.03 * norm_uncertainty_boost
    )
    
    score = np.where(is_urgent, score, score + non_urgent_contrib)
    score = np.clip(score, -1000000000000.0, 1000000000000.0)
    score = np.nan_to_num(score, nan=1000000000000.0, posinf=1000000000000.0, neginf=-1000000000000.0)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
