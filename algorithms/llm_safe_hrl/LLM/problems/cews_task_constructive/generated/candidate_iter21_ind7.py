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
    v2 priority rule: Strict urgency exclusivity + tightened critical-path gating + 
                      exponential uncertainty decay + robust fairness + deadline-proximity continuity.

    Key improvements over v1:
    - Urgency is *exclusive*: no sigmoid mixing — urgent tasks (slack <= 0) get hard -1e12, non-urgent use smooth proximity only.
    - Critical-path energy penalty now requires stricter tightness: rel_slack <= 0.1 (not 0.3), avoiding premature penalization.
    - Uncertainty boost decays exponentially with slack: exp(-max(0, slack)/tau), preserving urgency dominance while rewarding near-deadline risk awareness.
    - Fairness uses *normalized wait-per-work* (not raw wait time) with dynamic percentile threshold and robust z-score fallback.
    - All normalizations use degenerate-safe minmax with explicit nan/inf cleanup before percentile computation.
    - Final weights rebalanced: urgency dominates (0.45), critical latency (0.2), gated energy (0.15), 
      normalized fairness (0.1), uncertainty (0.07), remaining work (0.03).
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
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        # Ensure finite values for percentile computation
        x_finite = x[np.isfinite(x)]
        if x_finite.size == 0:
            return np.zeros_like(x)
        p01 = np.percentile(x_finite, 1.0, method='midpoint')
        p99 = np.percentile(x_finite, 99.0, method='midpoint')
        x_clipped = np.clip(x, p01, p99)
        x_min = np.min(x_clipped)
        x_max = np.max(x_clipped)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x_clipped - x_min) / (x_max - x_min + eps)

    # === STRICT URGENCY EXCLUSIVITY ===
    is_urgent = (slack <= 0.0).astype(float)
    # Non-urgent tasks use continuous deadline proximity: exp(-max(0, slack)/tau), tau=10s → strong pull near deadline
    tau = 10.0
    proximity_bias = np.exp(-np.maximum(0.0, slack) / tau)  # [0,1], decays smoothly from 1→0 as slack increases

    # === CRITICAL LATENCY ===
    duration = min_exec_time + min_comm_time + eps
    rel_slack = np.divide(slack, duration, out=np.zeros_like(slack), where=duration != 0)
    rel_slack = np.where(np.isfinite(rel_slack), rel_slack, 0.0)
    critical_latency_raw = duration * (1.0 + 0.7 * robust_minmax_norm(upward_rank))
    norm_critical_latency = robust_minmax_norm(critical_latency_raw)

    # === GATED ENERGY PENALTY (TIGHTENED THRESHOLD) ===
    energy_density = np.divide(min_incremental_energy, duration + eps, out=np.zeros_like(min_incremental_energy), where=duration + eps != 0)
    rank_threshold = np.percentile(upward_rank, 75.0) + eps
    # Tighter gate: only penalize when *very* tight (rel_slack <= 0.1) AND high-rank
    tight_slack_mask = (rel_slack <= 0.1).astype(float)
    high_rank_mask = (upward_rank > rank_threshold).astype(float)
    energy_penalty_mask = tight_slack_mask * high_rank_mask
    norm_energy_density = robust_minmax_norm(energy_density)
    energy_penalty = norm_energy_density * energy_penalty_mask

    # === ADAPTIVE FAIRNESS (WAIT-PER-WORK BASED, ROBUST) ===
    wait_per_work = np.divide(ready_wait_time, remaining_work + eps, out=np.zeros_like(ready_wait_time), where=remaining_work + eps != 0)
    wait_per_work = np.nan_to_num(wait_per_work, nan=0.0, posinf=0.0, neginf=0.0)
    # Dynamic threshold at 10th percentile of finite wait_per_work
    wpw_finite = wait_per_work[np.isfinite(wait_per_work)]
    work_threshold = np.percentile(wpw_finite, 10.0) + eps if wpw_finite.size > 0 else eps
    wait_gate = (wait_per_work >= work_threshold).astype(float)
    norm_wait_per_work = robust_minmax_norm(wait_per_work)
    wait_penalty = (1.0 - is_urgent) * norm_wait_per_work * wait_gate

    # === UNCERTAINTY BOOST WITH EXPONENTIAL DECAY (NOT SIGMOID) ===
    # Boost only for non-urgent & near-deadline: decays exponentially with slack (not relative slack)
    uncertainty_boost = uncertainty * proximity_bias  # proximity_bias already in [0,1]
    norm_uncertainty_boost = robust_minmax_norm(uncertainty_boost)

    # === REMAINING WORK NORMALIZATION ===
    norm_remaining_work = robust_minmax_norm(remaining_work)

    # === FINAL SCORE ASSEMBLY ===
    # Base score for non-urgent tasks
    base_score = np.full(N, 1.0, dtype=float)
    # Start with urgent minimum
    score = np.where(is_urgent, -1000000000000.0, base_score)
    # Add weighted components *only* to non-urgent
    score = np.where(
        is_urgent,
        score,
        score + 
        0.2 * norm_critical_latency + 
        0.15 * energy_penalty + 
        0.1 * wait_penalty + 
        0.07 * norm_uncertainty_boost + 
        0.03 * norm_remaining_work
    )
    # Apply proximity bias *only* to non-urgent: higher proximity → lower score (more urgent-like)
    score = np.where(
        is_urgent,
        score,
        score * (1.0 - proximity_bias) + (-1000000000000.0) * proximity_bias
    )
    # Clip and sanitize
    score = np.clip(score, -1000000000000.0, 1000000000000.0)
    score = np.nan_to_num(score, nan=1000000000000.0, posinf=1000000000000.0, neginf=-1000000000000.0)
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
