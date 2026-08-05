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
    v2 priority rule: Hard-deadline-respecting urgency-energy-criticality fusion with monotonic risk coupling.
    
    Key improvements over v1:
    - Restores strict hard-urgency gating: slack <= 0 → -1e12 (no relaxation to 0.1)
    - Removes fragile urgent/non-urgent energy stratification → unified robust energy density normalization
    - Replaces exp-based criticality decay with *monotonic slack-gated rank scaling*: 
      high upward_rank only amplified when rel_slack <= 0.2 (tight), else attenuated → preserves critical-path focus
    - Eliminates work_bias term (redundant with wait_penalty + remaining_work norm) → cleaner signal
    - Introduces *slack-aware uncertainty amplification*: uncertainty boosted only if (rel_slack < 0.5 AND upward_rank > 75th percentile)
    - Adds deadline proximity term: 1.0 / (max(0, slack) + eps) for non-urgent tasks → smooth urgency gradient
    - All normalizations use deterministic 1%-99% clipping + min-max with degenerate fallback
    - Strict zero/Nan/inf guarding; fully deterministic and side-effect free
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
        p01 = np.percentile(x, 1.0)
        p99 = np.percentile(x, 99.0)
        x_clipped = np.clip(x, p01, p99)
        x_min = np.min(x_clipped)
        x_max = np.max(x_clipped)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x_clipped - x_min) / (x_max - x_min + eps)
    
    # Hard urgency enforcement: absolute priority for violated or imminent deadlines
    is_urgent = (slack <= 0.0).astype(float)
    
    # Duration base (execution + communication) with safety
    duration = min_exec_time + min_comm_time + eps
    
    # Relative slack: slack / duration; clipped and sanitized
    rel_slack = np.divide(slack, duration, out=np.zeros_like(slack), where=duration != 0)
    rel_slack = np.where(np.isfinite(rel_slack), rel_slack, 0.0)
    
    # Criticality gating: amplify upward_rank only when slack is tight (rel_slack <= 0.2), else attenuate
    tight_slack_gate = (rel_slack <= 0.2).astype(float)
    gated_upward_rank = upward_rank * tight_slack_gate + upward_rank * 0.3 * (1.0 - tight_slack_gate)
    norm_gated_rank = robust_minmax_norm(gated_upward_rank)
    
    # Critical latency: duration weighted by gated rank importance
    critical_latency_raw = duration * (1.0 + 0.8 * norm_gated_rank)
    norm_critical_latency = robust_minmax_norm(critical_latency_raw)
    
    # Energy density: incremental energy per unit duration, robustly normalized
    energy_density = np.divide(min_incremental_energy, duration + eps, 
                                out=np.zeros_like(min_incremental_energy), 
                                where=duration + eps != 0)
    norm_energy_density = robust_minmax_norm(energy_density)
    
    # Energy penalty: applied only on critical path (high rank) AND tight slack
    rank_threshold = np.percentile(upward_rank, 75.0) + eps
    high_rank_mask = (upward_rank > rank_threshold).astype(float)
    energy_penalty_mask = high_rank_mask * tight_slack_gate
    energy_penalty = norm_energy_density * energy_penalty_mask
    
    # Starvation guard: wait-time penalty only for non-urgent & sufficiently large work
    work_threshold = np.percentile(remaining_work, 5.0) + eps
    wait_gate = (remaining_work >= work_threshold).astype(float)
    norm_wait_time = robust_minmax_norm(ready_wait_time)
    wait_penalty = (1.0 - is_urgent) * norm_wait_time * wait_gate
    
    # Uncertainty amplification: only when both slack is tight AND rank is high
    uncertainty_gate = high_rank_mask * (rel_slack < 0.5).astype(float)
    uncertainty_boost = uncertainty * uncertainty_gate
    norm_uncertainty_boost = robust_minmax_norm(uncertainty_boost)
    
    # Deadline proximity gradient for non-urgent tasks: higher score boost as slack shrinks
    proximity_boost = np.divide(1.0, np.maximum(0.0, slack) + eps, 
                                out=np.zeros_like(slack), where=slack > 0.0)
    norm_proximity = robust_minmax_norm(proximity_boost)
    
    # Base score structure
    base_score = np.full(N, 1.0, dtype=float)
    
    # Assemble final score: smallest = highest priority
    score = np.where(is_urgent, -1000000000000.0, base_score)
    score = np.where(
        is_urgent,
        score,
        score + 
        0.27 * norm_critical_latency + 
        0.22 * energy_penalty + 
        0.15 * wait_penalty + 
        0.12 * norm_uncertainty_boost + 
        0.11 * norm_proximity + 
        0.08 * (1.0 - robust_minmax_norm(slack)) + 
        0.05 * robust_minmax_norm(remaining_work)
    )
    
    # Final clamping and NaN/inf cleanup
    score = np.clip(score, -1000000000000.0, 1000000000000.0)
    score = np.nan_to_num(score, nan=1000000000000.0, posinf=1000000000000.0, neginf=-1000000000000.0)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
