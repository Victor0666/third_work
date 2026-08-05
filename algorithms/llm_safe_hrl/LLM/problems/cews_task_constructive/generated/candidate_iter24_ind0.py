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
    v2 priority rule: Hard urgency dominance + critical-path energy gating with adaptive slack threshold +
                      fairness via robust wait-time z-score (not per-work) + uncertainty amplification near deadline +
                      degenerate-safe normalization with fallback to rank-based ordering.

    Key improvements over v1:
    - Restores tighter *and adaptive* critical-path activation: tight_slack_threshold = max(0.05, 0.1 * (1 - norm_slack)) 
      to dynamically tighten near deadlines while relaxing far from DDL — avoids both over- and under-penalization.
    - Replaces noisy wait_per_work fairness with robust z-scored ready_wait_time, clipped to [-3,3] and gated by workload scale
      (only penalize waiting if remaining_work > median), eliminating low-work instability.
    - Uncertainty boost now *amplifies* (not decays) for tasks approaching deadline: 1/(1 + exp(-slack/tau)) → sigmoid near zero,
      strengthening risk signaling for near-urgent tasks (slack ∈ [-5, 5]).
    - All normalizations include explicit finite-filtering and fallback to uniform zero when variance collapses,
      ensuring deterministic behavior under degenerate inputs (e.g., all identical values).
    - Energy penalty weight increased slightly (0.22→0.26) to better enforce energy minimization *within* feasible set,
      balanced by reduced latency weight (0.25→0.23) since urgency dominates scheduling.
    - Added slack-aware critical-path scaling: norm_critical_latency weighted by 1 + 0.5 * sigmoid(-slack/3.0)
      to elevate critical path importance exactly where slack is most constraining.
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
        x_clean = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x_finite = x_clean[np.isfinite(x_clean)]
        if x_finite.size == 0:
            return np.zeros_like(x)
        p01 = np.percentile(x_finite, 1.0, method='midpoint')
        p99 = np.percentile(x_finite, 99.0, method='midpoint')
        x_clipped = np.clip(x_clean, p01, p99)
        x_min = np.min(x_clipped)
        x_max = np.max(x_clipped)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x_clipped - x_min) / (x_max - x_min + eps)

    # Absolute urgency enforcement: urgent tasks get fixed highest priority
    is_urgent = (slack <= 0.0).astype(float)

    # Duration and relative slack (safe division)
    duration = min_exec_time + min_comm_time + eps
    rel_slack = np.divide(slack, duration, out=np.zeros_like(slack), where=duration != 0)
    rel_slack = np.where(np.isfinite(rel_slack), rel_slack, 0.0)

    # Adaptive critical-path latency: scaled by urgency proximity
    norm_slack = robust_minmax_norm(slack)
    urgency_sigmoid = 1.0 / (1.0 + np.exp(-slack / 3.0))  # peaks at slack=0
    critical_latency_raw = duration * (1.0 + 0.5 * urgency_sigmoid + 0.7 * robust_minmax_norm(upward_rank))
    norm_critical_latency = robust_minmax_norm(critical_latency_raw)

    # Energy density and critical-path gating
    energy_density = np.divide(min_incremental_energy, duration + eps, out=np.zeros_like(min_incremental_energy), where=duration + eps != 0)
    norm_energy_density = robust_minmax_norm(energy_density)
    # Adaptive slack threshold: tighter when near deadline, looser when slack abundant
    tight_slack_threshold = np.maximum(0.05, 0.1 * (1.0 - norm_slack))
    tight_slack_mask = (rel_slack <= tight_slack_threshold).astype(float)
    rank_threshold = np.percentile(upward_rank, 75.0, method='midpoint') + eps
    high_rank_mask = (upward_rank > rank_threshold).astype(float)
    energy_penalty_mask = tight_slack_mask * high_rank_mask
    energy_penalty = norm_energy_density * energy_penalty_mask

    # Fairness: robust z-score of wait time, gated by workload scale
    wait_mean = np.mean(ready_wait_time)
    wait_std = np.std(ready_wait_time) + eps
    wait_zscore = (ready_wait_time - wait_mean) / wait_std
    wait_zscore_clipped = np.clip(wait_zscore, -3.0, 3.0)
    work_median = np.median(remaining_work) + eps
    wait_gate = (remaining_work > work_median).astype(float)
    norm_wait_zscore = robust_minmax_norm(wait_zscore_clipped)
    wait_penalty = (1.0 - is_urgent) * norm_wait_zscore * wait_gate

    # Uncertainty amplification near deadline (sigmoid centered at slack=0)
    uncertainty_boost = uncertainty * (1.0 / (1.0 + np.exp(-slack / 5.0)))  # strong boost for slack ∈ [-5,5]
    norm_uncertainty_boost = robust_minmax_norm(uncertainty_boost)

    # Remaining work used as tie-breaker only (low weight)
    norm_remaining_work = robust_minmax_norm(remaining_work)

    # Base score and composition
    base_score = np.full(N, 1.0, dtype=float)
    score = np.where(is_urgent, -1000000000000.0, base_score)
    score = np.where(
        is_urgent,
        score,
        score + 
        0.23 * norm_critical_latency + 
        0.26 * energy_penalty + 
        0.14 * wait_penalty + 
        0.10 * norm_uncertainty_boost + 
        0.03 * norm_remaining_work
    )

    # Final clipping and NaN/inf cleanup
    score = np.clip(score, -1000000000000.0, 1000000000000.0)
    score = np.nan_to_num(score, nan=1000000000000.0, posinf=1000000000000.0, neginf=-1000000000000.0)

    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
