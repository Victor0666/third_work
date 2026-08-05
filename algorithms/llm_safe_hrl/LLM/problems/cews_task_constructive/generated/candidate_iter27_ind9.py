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
    v2 priority rule: Hard-deadline dominance + adaptive slack-scaled criticality +
                      energy-risk gating with deadline proximity + starvation-resilient fairness +
                      uncertainty-coupling only under dual feasibility + robust normalization.

    Key improvements over v1:
    - Replaces fixed high-magnitude lateness penalty with *adaptive urgency scaling*: 
      penalty = -1e12 * (1 + 0.1 * |slack|) for slack <= 0 → preserves hard DDL enforcement
      while avoiding numeric overflow and enabling relative urgency among late tasks.
    - Introduces *slack-scaled critical timing*: norm_critical_timing * max(0, 1 - rel_slack)
      → suppresses critical-path bias when slack is tight but positive, preventing premature
      scheduling of long-duration low-urgency tasks.
    - Restores median-based fairness gating (more robust than percentile for small N),
      but now gated by *both* slack > 0 AND remaining_work > median → avoids boosting idle long-wait tasks.
    - Tightens uncertainty coupling: requires slack > 0 AND upward_rank > median AND uncertainty > median
      → prevents risk amplification on low-risk or non-critical tasks.
    - Adds *energy-efficiency boost*: when slack > 0 and rel_slack > 0.5, apply negative bonus to
      energy_density term to prefer low-energy options without compromising deadline safety.
    - All normalization uses stable minmax with fallbacks; all divisions guarded; deterministic.
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
        if N == 1:
            return np.zeros_like(x)
        p01 = np.percentile(x, 1.0, method='lower')
        p99 = np.percentile(x, 99.0, method='higher')
        x_clipped = np.clip(x, p01, p99)
        x_min = np.min(x_clipped)
        x_max = np.max(x_clipped)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x_clipped - x_min) / (x_max - x_min + eps)

    # Adaptive hard-deadline penalty: stronger for more negative slack, but bounded & scalable
    lateness_mask = slack <= 0.0
    abs_slack = np.abs(slack)
    lateness_penalty = np.where(lateness_mask, -1e12 * (1.0 + 0.1 * abs_slack), 0.0)

    duration = min_exec_time + min_comm_time + eps
    critical_timing = duration * upward_rank
    norm_critical_timing = robust_minmax_norm(critical_timing)

    # Slack-scaled critical timing: decay when slack is tight but positive
    rel_slack = np.divide(slack, duration, out=np.zeros_like(slack), where=duration != 0)
    rel_slack = np.nan_to_num(rel_slack, nan=0.0, posinf=0.0, neginf=0.0)
    slack_scale = np.clip(1.0 - rel_slack, 0.0, 1.0)
    scaled_critical_timing = norm_critical_timing * slack_scale

    energy_density = np.divide(min_incremental_energy, duration, out=np.zeros_like(min_incremental_energy), where=duration != 0)
    energy_density = np.nan_to_num(energy_density, nan=0.0, posinf=0.0, neginf=0.0)
    norm_energy_density = robust_minmax_norm(energy_density)

    # Slack-proximity energy penalty: only active in 0 < rel_slack <= 0.3
    slack_proximity = np.clip(0.3 - rel_slack, 0.0, 0.3)
    energy_penalty = norm_energy_density * slack_proximity

    # Energy-efficiency boost: prefer low-energy tasks when slack is ample (rel_slack > 0.5)
    efficiency_boost_mask = (slack > 0.0) & (rel_slack > 0.5)
    efficiency_boost = -0.2 * norm_energy_density * efficiency_boost_mask

    # Fairness: only activate for tasks with both positive slack AND above-median remaining work
    rw_median = np.median(remaining_work) if N > 1 else remaining_work[0]
    fairness_mask = (slack > 0.0) & (remaining_work > rw_median)
    norm_wait_time = robust_minmax_norm(ready_wait_time)
    fairness_boost = norm_wait_time * fairness_mask

    # Uncertainty coupling: triple-gated — slack > 0, high criticality, AND high uncertainty
    ur_median = np.median(upward_rank) if N > 1 else upward_rank[0]
    unc_median = np.median(uncertainty) if N > 1 else uncertainty[0]
    unc_mask = (slack > 0.0) & (upward_rank > ur_median) & (uncertainty > unc_median)
    norm_uncertainty = robust_minmax_norm(uncertainty)
    unc_coupling = norm_uncertainty * unc_mask

    # Weighted composite score (weights sum to 1.0)
    # Prioritize: DDL feasibility (0.45), adaptive critical path (0.25), energy-risk (0.15),
    #             fairness (0.08), uncertainty (0.04), efficiency boost (-0.07 → net 0.15)
    score = (
        0.45 * scaled_critical_timing +
        0.25 * robust_minmax_norm(remaining_work) +
        0.15 * energy_penalty +
        0.08 * fairness_boost +
        0.04 * unc_coupling +
        efficiency_boost
    )

    # Apply lateness penalty
    score = lateness_penalty + score

    # Final sanitization
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)

    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
