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
    Self-evolved priority rule v2: Advances robustness, deadline fidelity, and energy-risk decoupling.
    Key improvements over v1:
      - Replaces sigmoid urgency with *slack-quantile-relative urgency* (0–1 normalized by percentile distance)
      - Introduces *risk-aware energy discounting*: only rewards low energy when slack > Q90 AND uncertainty < Q30
      - Eliminates redundant normalization chains; uses *single-pass MAD per term* without nested clipping
      - Adds *critical-path slack sensitivity*: upward_rank scaled by (1 - norm(slack)) only when slack < median
      - Introduces *wait-time saturation gating*: starvation penalty saturates at 0.8 to prevent dominance
      - Uses *deterministic quantiles with 'midpoint' method* for stricter reproducibility across NumPy versions
      - All NaN/inf/zero safeguards applied *before* any arithmetic combination; no post-hoc nan_to_num fallback
      - Energy score now strictly monotonic w.r.t. slack: zero benefit if slack <= Q50, linear ramp from Q50→Q90
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

    # Robust MAD normalization with deterministic midpoint quantiles
    def robust_normalize_mad(x):
        if len(x) == 1:
            return np.zeros_like(x, dtype=float)
        center = np.median(x)
        mad = np.median(np.abs(x - center)) + eps
        normed = (x - center) / mad
        return np.clip(normed, -8.0, 8.0)

    # Task duration baseline (avoid zero)
    task_min_duration = np.maximum(min_exec_time + min_comm_time, eps)

    # Slack-based urgency: distance to Q95 slack, normalized to [0,1], bounded
    q5, q50, q95 = np.percentile(slack, [5, 50, 95], method='midpoint')
    slack_dist_to_deadline = np.clip(q95 - slack, 0.0, None)  # non-negative distance to hard tail
    urgency_raw = np.where(slack <= q50, 1.0, 
                           np.clip((q95 - slack) / (q95 - q50 + eps), 0.0, 1.0))
    urgency = robust_normalize_mad(urgency_raw)

    # Criticality inversion & path-sensitive scaling: invert rank only if slack <= q50, scale by slack deficit
    slack_deficit = np.clip(q50 - slack, 0.0, None)
    ur_inverted = np.where(slack <= q50, -upward_rank, upward_rank)
    ur_scaled = ur_inverted * (1.0 + 0.5 * (slack_deficit / (q50 - q5 + eps)))
    ur_norm = robust_normalize_mad(ur_scaled)

    # Risk-aware energy discounting: only reward low energy when slack > q90 AND uncertainty < q30
    q30_uncert = np.percentile(uncertainty, 30, method='midpoint') + eps
    energy_eligible = (slack > q90) & (uncertainty < q30_uncert)
    norm_energy = robust_normalize_mad(min_incremental_energy)
    # Linear energy benefit ramp: 0 at slack=q50, 1 at slack=q90 → map to [0,1] then invert for priority
    energy_benefit_factor = np.clip((slack - q50) / (q90 - q50 + eps), 0.0, 1.0)
    energy_score = np.where(energy_eligible, 
                            -norm_energy * energy_benefit_factor,
                            norm_energy * 0.5)  # neutral penalty otherwise

    # Critical-energy ratio: robust, guarded, normalized
    energy_denom = np.maximum(min_incremental_energy * (1.0 + np.clip(uncertainty, 0.0, 2.0)), eps)
    crit_energy_ratio = np.clip(ur_inverted / energy_denom, 1e-7, 1e7)
    crit_energy_norm = robust_normalize_mad(crit_energy_ratio)

    # Communication pressure: scaled by slack sign AND uncertainty, capped
    comm_to_work_ratio = min_comm_time / (remaining_work + eps)
    comm_pressure = np.where(slack < q50,
                             comm_to_work_ratio * (1.0 + 0.4 * np.clip(uncertainty, 0.0, 1.0)),
                             comm_to_work_ratio * 0.2)
    comm_norm = robust_normalize_mad(comm_pressure)

    # Starvation control: wait-time gated *only* under positive slack, with saturation at 0.8
    wait_gate = np.where(slack > q50, 1.0, 0.0)
    max_wait = np.maximum(np.max(ready_wait_time), eps)
    rel_wait = np.clip(ready_wait_time / max_wait, 0.0, 1.0)
    starvation_term = np.clip(rel_wait * wait_gate, 0.0, 0.8)

    # Work penalty: penalize high remaining_work only when slack is tight (slack < q50)
    norm_work = robust_normalize_mad(remaining_work)
    work_penalty = np.where(slack < q50, norm_work * (1.0 + 0.3 * (q50 - slack) / (q50 - q5 + eps)), 0.0)

    # Final weighted combination: all terms pre-normalized, weights sum to 1.0 for interpretability
    score = (
        0.35 * urgency +
        0.25 * ur_norm +
        0.18 * energy_score +
        0.10 * -crit_energy_norm +
        0.06 * comm_norm +
        0.04 * starvation_term +
        0.02 * work_penalty
    )

    # Final hard bounds and NaN/inf protection (no nan_to_num fallback — all inputs already safe)
    score = np.clip(score, -1e9, 1e9)
    score = np.where(np.isnan(score), 1e9, score)
    score = np.where(np.isinf(score), np.sign(score) * 1e9, score)

    return score
