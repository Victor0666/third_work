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
    v2 priority rule: Hard binary urgency + sparse critical-path energy gating + 
    robust wait fairness + uncertainty-aware slack coupling + monotonic normalization.
    
    Key improvements:
    - Restores strict hard deadline enforcement: is_urgent = (slack <= 0) — no percentile threshold.
    - Replaces sigmoid energy gate with crisp AND mask: high-rank ∧ tight-slack (rel_slack <= 0.2).
    - Uses robust_minmax_norm exclusively (no z-score hybrids) for monotonicity and DDL compliance.
    - Introduces *normalized progress pressure*: upward_rank / (duration + eps), scaled by slack proximity.
    - Uncertainty coupling now gated by both urgency and criticality: only active when slack <= 0 OR upward_rank > median.
    - All components clipped, normalized, and fused with deterministic weights summing to 1.0.
    - Eliminates non-monotonic functions (sigmoid, z-score) to preserve ordering under fuzzy uncertainty.
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
        p01 = np.percentile(x, 1.0, method='midpoint')
        p99 = np.percentile(x, 99.0, method='midpoint')
        x_clipped = np.clip(x, p01, p99)
        x_min = np.min(x_clipped)
        x_max = np.max(x_clipped)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x_clipped - x_min) / (x_max - x_min + eps)

    # Hard urgency: binary, zero-tolerance deadline violation detection
    is_urgent = (slack <= 0.0).astype(np.float64)
    urgency_score = np.full(N, -1e12, dtype=np.float64)

    # Duration and derived metrics
    duration = min_exec_time + min_comm_time + eps
    rel_slack = np.divide(slack, duration, out=np.zeros_like(slack), where=duration != 0)
    rel_slack = np.where(np.isfinite(rel_slack), rel_slack, 0.0)

    # Critical path latency: duration weighted by upward rank importance
    critical_latency_raw = duration * (1.0 + 0.7 * robust_minmax_norm(upward_rank))
    norm_critical_latency = robust_minmax_norm(critical_latency_raw)

    # Energy density: marginal energy per time unit — penalize only on critical+tight paths
    energy_density = np.divide(min_incremental_energy, duration + eps, out=np.zeros_like(min_incremental_energy), where=duration + eps != 0)
    energy_density = np.where(np.isfinite(energy_density), energy_density, 0.0)
    norm_energy_density = robust_minmax_norm(energy_density)

    # Sparse energy penalty mask: only high criticality AND tight slack (rel_slack <= 0.2)
    median_upward = np.median(upward_rank) + eps
    high_rank_mask = (upward_rank > median_upward).astype(np.float64)
    tight_slack_mask = (rel_slack <= 0.2).astype(np.float64)
    energy_penalty_mask = high_rank_mask * tight_slack_mask
    energy_penalty = norm_energy_density * energy_penalty_mask

    # Wait fairness: normalized wait time, gated by non-urgency and sufficient work
    work_threshold = np.percentile(remaining_work, 5.0, method='midpoint') + eps
    wait_gate = (remaining_work >= work_threshold).astype(np.float64)
    norm_wait_time = robust_minmax_norm(ready_wait_time)
    wait_penalty = (1.0 - is_urgent) * norm_wait_time * wait_gate

    # Progress pressure: upward_rank / duration, scaled by slack proximity to deadline
    progress_pressure = np.divide(upward_rank, duration, out=np.zeros_like(upward_rank), where=duration != 0)
    # Scale by how close slack is to zero (higher weight near deadline)
    slack_proximity = np.clip(-slack / (np.abs(slack) + eps), 0.0, 1.0)
    norm_progress_pressure = robust_minmax_norm(progress_pressure) * slack_proximity

    # Uncertainty coupling: only activated for urgent OR high-criticality tasks
    uncertainty_active = np.maximum(is_urgent, high_rank_mask)
    abs_rel_slack = np.abs(rel_slack) + eps
    slack_sensitivity = np.clip(1.0 / abs_rel_slack, 0.1, 10.0)
    rank_sensitivity = robust_minmax_norm(upward_rank)
    uncertainty_boost = uncertainty * slack_sensitivity * rank_sensitivity * uncertainty_active
    norm_uncertainty_boost = robust_minmax_norm(uncertainty_boost)

    # Work bias: slight preference for larger workloads to amortize setup cost
    norm_remaining_work = robust_minmax_norm(remaining_work)

    # Weighted fusion: deterministic, monotonic, sum-to-1.0
    score = (
        0.38 * (1.0 - robust_minmax_norm(np.clip(-slack, 0.0, np.inf))) +  # urgency proximity
        0.25 * energy_penalty +                                          # critical-path energy
        0.14 * wait_penalty +                                            # fairness
        0.10 * norm_uncertainty_boost +                                  # risk-aware coupling
        0.08 * (1.0 - norm_progress_pressure) +                          # progress pressure (higher = lower priority)
        0.05 * norm_remaining_work                                       # work bias
    )

    # Apply hard urgency override
    score = np.where(is_urgent, urgency_score, score)

    # Final clipping and NaN/inf cleanup
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)

    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
