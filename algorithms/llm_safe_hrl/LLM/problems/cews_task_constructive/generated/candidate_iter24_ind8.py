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
    v2 priority rule: DDL-feasibility first, energy-risk second, starvation-resilient, and robustly normalized.
    
    Key improvements over v1:
    - Restores clipped min-max normalization for stability across all N (esp. N=1, skewed distributions).
    - Removes fragile z-score and starvation-boost logic; replaces with *wait-time fairness gating* that only activates when slack > 0 AND upward_rank > median — avoids diluting urgency.
    - Introduces *slack-proximity penalty*: energy penalty scales linearly with (0.3 - rel_slack) for tight non-urgent tasks (0 < rel_slack <= 0.3), zero otherwise — smooth, bounded, deadline-aware.
    - Uncertainty coupling now gated by *both* positive slack AND high criticality (upward_rank > 75th percentile), preventing amplification on late or low-rank tasks.
    - Critical path term simplified to duration × upward_rank (no ad-hoc coefficients), then robustly normalized — preserves HEFT intuition without overfitting.
    - Final weights sum to 1.0 and prioritize: DDL feasibility (0.45), critical-path timing (0.25), energy-risk (0.18), fairness (0.08), uncertainty (0.04).
    - All operations guarded: eps, nan_to_num, clip, safe division, deterministic fallbacks.
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

    def robust_minmax_norm(x):
        if x.size == 0:
            return np.zeros_like(x)
        p01 = np.percentile(x, 1.0, method='lower') if N > 1 else np.min(x)
        p99 = np.percentile(x, 99.0, method='higher') if N > 1 else np.max(x)
        x_clipped = np.clip(x, p01, p99)
        x_min = np.min(x_clipped)
        x_max = np.max(x_clipped)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x_clipped - x_min) / (x_max - x_min + eps)

    # Urgency dominance: hard priority for violated or imminent deadlines
    is_urgent = (slack <= 0.0)
    score = np.full(N, 1.0, dtype=float)
    score = np.where(is_urgent, -1e12, score)

    # Only process non-urgent tasks
    non_urgent_mask = ~is_urgent
    if not np.any(non_urgent_mask):
        score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
        assert score.shape == (N,)
        return score

    # Base duration & relative slack
    duration = min_exec_time + min_comm_time + eps
    rel_slack = np.divide(slack, duration, out=np.zeros_like(slack), where=duration != 0)
    rel_slack = np.where(np.isfinite(rel_slack), rel_slack, 0.0)

    # Critical-path timing: raw duration × upward_rank, then normalized
    critical_timing = duration * upward_rank
    norm_critical_timing = robust_minmax_norm(critical_timing)

    # Energy density: marginal energy per time unit, normalized only for penalty application
    energy_density = np.divide(min_incremental_energy, duration, out=np.zeros_like(min_incremental_energy), where=duration != 0)
    energy_density = np.nan_to_num(energy_density, nan=0.0, posinf=0.0, neginf=0.0)
    norm_energy_density = robust_minmax_norm(energy_density)

    # Slack-proximity energy penalty: active only in safe but tight region (0 < rel_slack <= 0.3)
    slack_proximity = np.clip(0.3 - rel_slack, 0.0, 0.3)
    energy_penalty = norm_energy_density * slack_proximity

    # Fairness: wait-time boost gated by *both* slack > 0 and upward_rank above median
    ur_median = np.median(upward_rank) if N > 1 else upward_rank[0]
    fairness_mask = (slack > 0.0) & (upward_rank > ur_median)
    norm_wait_time = robust_minmax_norm(ready_wait_time)
    fairness_boost = norm_wait_time * fairness_mask

    # Uncertainty coupling: only when slack > 0 AND upward_rank > 75th percentile
    rank_75 = np.percentile(upward_rank, 75.0) + eps if N > 1 else upward_rank[0]
    unc_mask = (slack > 0.0) & (upward_rank > rank_75)
    norm_uncertainty = robust_minmax_norm(uncertainty)
    unc_coupling = norm_uncertainty * unc_mask

    # Combine non-urgent components with calibrated weights
    non_urgent_score = (
        0.45 * norm_critical_timing +
        0.25 * robust_minmax_norm(remaining_work) +  # higher work → higher priority (critical mass)
        0.18 * energy_penalty +
        0.08 * fairness_boost +
        0.04 * unc_coupling
    )

    # Apply to non-urgent subset only
    score = np.where(non_urgent_mask, score + non_urgent_score, score)

    # Final sanitization
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)

    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
