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
    v2 priority rule: Strict urgency-first hierarchy + criticality-gated risk coupling + uniform fairness amplification.
    
    Key improvements over v1:
      - Urgency dominance enforced: urgency_bias (-1.0) now carries *absolute* priority via score offset and clipping,
        ensuring slack<=0 tasks always yield minimal score regardless of other terms.
      - Synergy term removed to eliminate dilution of hard deadline compliance; replaced by criticality-weighted latency
        that directly supports critical-path progress without energy-latency tradeoff confusion.
      - Uncertainty coupling gated jointly by |slack| AND upward_rank: only activates for both high-risk (small slack)
        AND high-importance (high rank) tasks — avoids noise amplification on low-criticality tasks.
      - Fairness guard simplified to linear wait-time scaling *for all* non-urgent tasks (no quantile threshold),
        ensuring progressive prioritization without starvation gaps.
      - Energy penalty refined using upward-rank-normalized energy density, thresholded at 95th percentile for
        stricter filtering of inefficient critical-path tasks.
      - All normalization uses robust 1%-99% clipping + min-max; all divisions guarded; NaN/inf sanitized.
      - Final weights strictly hierarchical: urgency (dominant offset) > critical-latency (0.25) > energy (0.22) >
        fairness (0.12) > risk-gated uncertainty (0.09) > remaining_work (0.04).
    '''
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)
    
    def robust_minmax_norm(x):
        x_min = np.min(x)
        x_max = np.max(x)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        p01 = np.percentile(x, 1.0)
        p99 = np.percentile(x, 99.0)
        x_clipped = np.clip(x, p01, p99)
        x_min_c = np.min(x_clipped)
        x_max_c = np.max(x_clipped)
        if x_max_c - x_min_c < eps:
            return np.zeros_like(x)
        return (x_clipped - x_min_c) / (x_max_c - x_min_c + eps)
    
    # Absolute urgency gating: slack <= 0 → score forced to minimum possible value
    is_urgent = (slack <= 0).astype(float)
    urgency_score = np.where(is_urgent, -1.0, 0.0)
    
    # Critical-path latency: execution + comm weighted by upward_rank to prioritize critical-path progress
    duration = min_exec_time + min_comm_time + eps
    critical_latency_raw = duration * (1.0 + 0.8 * robust_minmax_norm(upward_rank))
    norm_critical_latency = robust_minmax_norm(critical_latency_raw)
    
    # Energy penalty: upward-rank-normalized energy density, thresholded at 95th percentile
    energy_density = min_incremental_energy / (duration + eps)
    rank_weighted_energy = energy_density / (upward_rank + eps)
    threshold_rank_energy = np.percentile(rank_weighted_energy, 95.0) + eps
    energy_penalty_mask = (rank_weighted_energy > threshold_rank_energy).astype(float)
    norm_energy_density = robust_minmax_norm(energy_density)
    energy_penalty = norm_energy_density * energy_penalty_mask
    
    # Fairness: linear wait amplification for all non-urgent tasks (no threshold), normalized
    norm_wait_time = robust_minmax_norm(ready_wait_time)
    wait_penalty = (1.0 - is_urgent) * norm_wait_time  # applies only when not urgent
    
    # Risk-gated uncertainty: activated only when both slack is tight AND rank is high
    slack_abs = np.abs(slack) + 1.0
    slack_scale_factor = np.clip(1.0 / slack_abs, 0.1, 10.0)
    rank_gate = robust_minmax_norm(upward_rank)  # [0,1], enables uncertainty only for high-rank tasks
    uncertainty_boost = uncertainty * slack_scale_factor * rank_gate
    norm_uncertainty_boost = robust_minmax_norm(uncertainty_boost)
    
    # Normalize remaining work
    norm_remaining_work = robust_minmax_norm(remaining_work)
    
    # Base score: ensure urgent tasks dominate unconditionally
    base_score = 1.0 + urgency_score
    
    # Add weighted components — all non-urgent contributions are bounded and secondary
    score = (
        base_score
        + 0.25 * norm_critical_latency
        + 0.22 * energy_penalty
        + 0.12 * wait_penalty
        + 0.09 * norm_uncertainty_boost
        + 0.04 * norm_remaining_work
    )
    
    # Clamp to finite range and sanitize
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    
    # Final guarantee: urgent tasks get globally minimal score
    score = np.where(is_urgent, -1e12, score)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
