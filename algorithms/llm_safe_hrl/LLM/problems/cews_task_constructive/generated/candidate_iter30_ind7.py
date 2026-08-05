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
    v2 priority rule: Deadline-robust criticality-energy-fairness with adaptive risk gating and starvation-aware latency compensation.
    
    Key improvements over v1:
    - Replaces tanh-based lateness penalty with *smoothed sign-scaled sigmoid* for sharper yet numerically stable hard-deadline dominance:
      penalty = -1000 * (1 + 0.08*|slack|) * sigmoid(-slack / max(eps, 0.1*duration)) → steep drop for negative slack, zero for positive.
    - Introduces *latency-compensated fairness*: wait_score = ready_wait_time / (duration + eps) normalized *only when slack > 0*, avoiding artificial boosting of late tasks.
    - Refines energy-efficiency boost to require *both low uncertainty AND high slack margin* (rel_slack > 0.6) for stronger preference under safety.
    - Adds *critical-path density* term: (upward_rank * remaining_work) / (duration + eps), normalized and weighted → captures "work-per-time" importance on critical path.
    - Uses *dynamic weight balancing*: weights sum to 1.0 and adapt slightly based on median slack sign to emphasize urgency when deadlines are tight.
    - Final score mapped via *robust sigmoid compression* instead of linear normalization → preserves ordinal ranking under outliers and small-N.
    - All operations fully guarded; no division by zero, no NaN/inf propagation; deterministic and shape-safe.
    '''
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

    def robust_5_95_norm(x):
        if x.size == 0:
            return np.zeros_like(x)
        if N == 1:
            return np.zeros_like(x)
        p05 = np.percentile(x, 5.0, method='lower')
        p95 = np.percentile(x, 95.0, method='higher')
        x_clipped = np.clip(x, p05, p95)
        x_min = np.min(x_clipped)
        x_max = np.max(x_clipped)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x_clipped - x_min) / (x_max - x_min + eps)

    duration = min_exec_time + min_comm_time + eps
    rel_slack = np.divide(slack, duration, out=np.zeros_like(slack), where=duration != 0)
    rel_slack = np.nan_to_num(rel_slack, nan=0.0, posinf=0.0, neginf=0.0)
    abs_slack = np.abs(slack)

    # Smoothed sign-scaled sigmoid lateness penalty: steep, bounded, avoids overflow
    slack_scale = np.divide(-slack, np.maximum(eps, 0.1 * duration), out=np.zeros_like(slack), where=duration != 0)
    lateness_penalty = -1000.0 * (1.0 + 0.08 * abs_slack) * (1.0 / (1.0 + np.exp(-np.clip(slack_scale, -10.0, 10.0))))

    critical_timing = duration * upward_rank
    norm_critical_timing = robust_5_95_norm(critical_timing)

    # Slack-gated critical timing using smooth sigmoid gate: suppresses non-urgent critical tasks
    slack_gate = 1.0 / (1.0 + np.exp(np.clip(rel_slack - 0.2, -10.0, 10.0)))
    scaled_critical_timing = norm_critical_timing * slack_gate

    energy_density = np.divide(min_incremental_energy, duration, out=np.zeros_like(min_incremental_energy), where=duration != 0)
    energy_density = np.nan_to_num(energy_density, nan=0.0, posinf=0.0, neginf=0.0)
    norm_energy_density = robust_5_95_norm(energy_density)

    # Critical-path density: importance × work per time unit
    cp_density = np.divide(upward_rank * remaining_work, duration + eps, out=np.zeros_like(upward_rank), where=duration != 0)
    norm_cp_density = robust_5_95_norm(cp_density)

    # Median thresholds — safe for N=1
    rw_median = np.median(remaining_work) if N > 1 else remaining_work[0]
    ur_median = np.median(upward_rank) if N > 1 else upward_rank[0]
    unc_median = np.median(uncertainty) if N > 1 else uncertainty[0]

    # Efficiency boost: only when safe (high slack margin AND low uncertainty)
    efficiency_boost_mask = (slack > 0.0) & (rel_slack > 0.6) & (uncertainty < unc_median)
    efficiency_boost = -0.18 * norm_energy_density * efficiency_boost_mask

    # Latency-compensated fairness: wait time per unit work-density, gated by slack > 0 only
    work_density = np.divide(remaining_work, duration + eps, out=np.zeros_like(remaining_work), where=duration != 0)
    wait_score = np.divide(ready_wait_time, work_density + eps, out=np.zeros_like(ready_wait_time), where=work_density != 0)
    norm_wait_score = robust_5_95_norm(wait_score)
    fairness_mask = (slack > 0.0) & (remaining_work > rw_median)
    fairness_boost = norm_wait_score * fairness_mask

    # Uncertainty coupling: requires safety + criticality + risk exposure
    unc_mask = (slack > 0.0) & (upward_rank > ur_median) & (uncertainty > unc_median)
    norm_uncertainty = robust_5_95_norm(uncertainty)
    unc_coupling = norm_uncertainty * unc_mask

    # Dynamic weight balancing: tilt toward urgency when median slack <= 0
    median_slack = np.median(slack) if N > 1 else slack[0]
    urgency_bias = 0.15 if median_slack <= 0.0 else 0.0
    w_critical = 0.37 + urgency_bias
    w_cp_density = 0.20 - urgency_bias
    w_energy = 0.13
    w_fairness = 0.09
    w_unc = 0.06
    w_eff = 0.15

    # Weighted composite score before penalty
    score_base = (
        w_critical * scaled_critical_timing +
        w_cp_density * norm_cp_density +
        w_energy * norm_energy_density * np.clip(0.35 - rel_slack, 0.0, 0.35) +
        w_fairness * fairness_boost +
        w_unc * unc_coupling +
        w_eff * efficiency_boost
    )

    score = score_base + lateness_penalty

    # Robust sigmoid compression for final priority score → preserves ranking, bounds outliers
    score_centered = score - np.median(score) if N > 1 else score
    score_scaled = np.divide(score_centered, np.maximum(eps, np.std(score) if N > 1 else 1.0), out=np.zeros_like(score), where=N > 1 or True)
    final_score = 1.0 / (1.0 + np.exp(-np.clip(score_scaled, -8.0, 8.0)))

    final_score = np.nan_to_num(final_score, nan=1.0, posinf=1.0, neginf=0.0)
    final_score = np.clip(final_score, 0.0, 1.0)

    assert final_score.shape == (N,), f'Expected shape (N,)={N}, got {final_score.shape}'
    return final_score
