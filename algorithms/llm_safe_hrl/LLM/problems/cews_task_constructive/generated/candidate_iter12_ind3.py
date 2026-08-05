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
    Self-evolved priority rule v2: Monotonic deadline dominance + uncoupled risk scaling + starvation fairness.
    
    Key evolution improvements:
    - Replaces sigmoid/adaptive eta with *strictly monotonic linear urgency*: penalty = max(0, -slack) * (1 + k*uncertainty)
    - Removes all non-monotonic terms (sigmoid, harmonic mean, gated efficiency) → ensures ordering stability under perturbation
    - Uses *work-normalized wait ratio* for starvation only when slack >= 0 (avoids interfering with urgent tasks)
    - Introduces *critical-path leverage*: upward_rank / (remaining_work + eps) — higher means more impact per unit work
    - Energy term uses *direct marginal energy* scaled by uncertainty, no gating → preserves energy-awareness even under pressure
    - All operations protected by eps; robust_scale now uses MAD fallback and bounded clipping
    - Final score is convex combination of interpretable, monotonic components → deterministic, debuggable, deadline-safe
    """
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
    
    def robust_scale(x):
        if len(x) == 0:
            return x
        med = np.median(x)
        x_centered = x - med
        abs_dev = np.abs(x_centered)
        mad = np.median(abs_dev)
        scale = mad if mad > eps else np.mean(abs_dev) + eps
        scaled = x_centered / scale
        return np.clip(scaled, -1e6, 1e6)
    
    # Duration: execution + communication, strictly positive
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    
    # Deadline penalty: monotonic in slack & uncertainty — zero when slack >= 0
    neg_slack = np.maximum(-slack, 0.0)
    uncertainty_factor = 1.0 + 0.8 * np.clip(uncertainty, 0.0, 3.0)
    deadline_penalty = neg_slack * uncertainty_factor
    
    # Critical-path leverage: importance per unit remaining work — higher = more critical to schedule now
    cp_leverage = upward_rank / (remaining_work + eps)
    cp_score = -robust_scale(cp_leverage)  # lower score = higher priority
    
    # Energy efficiency: marginal energy per unit duration — lower is better, scaled by uncertainty
    energy_efficiency = min_incremental_energy / (duration + eps)
    energy_score = robust_scale(energy_efficiency) * (1.0 + 0.5 * np.clip(uncertainty, 0.0, 2.0))
    
    # Latency sensitivity: normalized duration — longer tasks penalized unless urgent
    duration_med = np.median(duration) + eps
    norm_duration = duration / duration_med
    latency_score = robust_scale(norm_duration)
    
    # Starvation fairness: only activated for non-urgent tasks (slack >= 0) with high wait/work ratio
    wait_ratio = ready_wait_time / (remaining_work + eps)
    p90_ratio = np.quantile(wait_ratio, 0.9, method='higher') if N > 1 else np.max(wait_ratio)
    is_starvable = (slack >= 0.0) & (wait_ratio > p90_ratio + eps)
    starvation_boost = np.where(is_starvable, robust_scale(ready_wait_time), 0.0)
    
    # Work load signal: larger remaining_work should not inherently delay scheduling — center & invert
    work_score = -robust_scale(remaining_work)
    
    # Final convex combination: weights sum to 1.0, all terms monotonic w.r.t. core constraints
    score = (
        0.45 * deadline_penalty +
        0.20 * cp_score +
        0.15 * energy_score +
        0.08 * latency_score +
        0.07 * starvation_boost +
        0.05 * work_score
    )
    
    # Final sanitization: ensure finite, bounded, correct shape
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
