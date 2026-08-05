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
    Self-evolved priority rule: deadline-hardness preserving urgency dominance,
    criticality-energy fairness under feasibility, and robust starvation control.
    
    Key improvements over v1:
      - Replaces dual urgency gates with *single, hardened sigmoid* centered at median_slack
        but steepened (scale=0.25*iqr) to enforce crisp deadline boundary without over-aggression.
      - Eliminates discontinuous slack-proportional work penalty; replaces with smooth,
        monotonic 'slack-sensitivity' term: exp(-max(0, -slack)/tau), tau adaptive to slack spread.
      - Criticality-energy term uses *normalized rank ratio* scaled only by urgency AND slack-feasibility,
        preventing suppression of high-criticality tasks even when slack is slightly negative.
      - Uncertainty boosting now strictly multiplicative on energy penalty *only when slack <= median_slack*,
        avoiding dilution near deadline while preserving signal integrity.
      - Starvation guard simplified: activated only for non-urgent (soft_urgency < 0.6) AND slack >= 0,
        using *relative wait percentile* (not fixed threshold) to adapt to workload burstiness.
      - All normalization uses robust_minmax_norm with fallback zeros; final score bounded and finite.
      - No component dominates >45% weight — ensures balanced tradeoff between DDL, energy, and fairness.
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
    
    def robust_minmax_norm(x):
        x_min, x_max = np.min(x), np.max(x)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x - x_min) / (x_max - x_min + eps)
    
    # Hardened urgency: steep sigmoid centered at median, scale proportional to IQR
    median_slack = np.median(slack)
    q1_slack, q3_slack = np.percentile(slack, [25, 75])
    iqr_slack = q3_slack - q1_slack + eps
    soft_urgency = 1.0 / (1.0 + np.exp(-(slack - median_slack) / (0.25 * iqr_slack + eps)))
    
    # Criticality-energy term: only active for feasible or mildly late tasks (slack > -1s), normalized and gated
    crit_per_energy = upward_rank / (min_incremental_energy + eps)
    norm_crit_per_energy = robust_minmax_norm(crit_per_energy)
    # Gate: active when slack > -1s OR urgency > 0.4 — prevents discarding critical late tasks
    crit_gate = np.where((slack > -1.0) | (soft_urgency > 0.4), 1.0, 0.0)
    crit_term = (1.0 - norm_crit_per_energy) * crit_gate
    
    # Energy penalty: base normalized energy + uncertainty boost *only when tight slack*
    norm_energy = robust_minmax_norm(min_incremental_energy)
    tight_slack_mask = (slack <= median_slack).astype(float)
    energy_penalty = norm_energy * (1.0 + 0.7 * uncertainty * tight_slack_mask)
    
    # Latency term: sum exec+comm, normalized
    total_latency = min_exec_time + min_comm_time
    norm_latency = robust_minmax_norm(total_latency)
    
    # Slack-sensitivity work penalty: smooth exponential decay for negative slack, zero otherwise
    tau = np.maximum(np.abs(median_slack), 1.0) + eps  # adaptive time constant
    slack_sensitivity = np.exp(-np.clip(np.maximum(-slack, 0.0), 0.0, 100.0) / tau)
    norm_work = robust_minmax_norm(remaining_work)
    work_penalty = norm_work * slack_sensitivity
    
    # Starvation guard: relative wait percentile, only for non-urgent & non-late
    wait_gate = np.where((soft_urgency < 0.6) & (slack >= 0), 1.0, 0.0)
    # Use percentile rank instead of fixed threshold — adapts to wait distribution
    wait_sorted = np.sort(ready_wait_time)
    wait_ranks = np.array([np.searchsorted(wait_sorted, w, side='right') for w in ready_wait_time])
    wait_percentile = wait_ranks / (N + eps)
    wait_boost = wait_percentile * wait_gate
    norm_wait_boost = robust_minmax_norm(wait_boost) if N > 0 else np.zeros(N)
    starvation_term = 1.0 - norm_wait_boost
    
    # Base urgency dominates but smoothly decays outside hard boundary
    base_urgency = 1.0 - soft_urgency
    
    # Final convex combination — weights sum to 1.0, no component > 0.45
    score = (
        0.45 * base_urgency +
        0.20 * energy_penalty +
        0.12 * norm_latency +
        0.10 * crit_term +
        0.06 * starvation_term +
        0.04 * work_penalty +
        0.03 * (1.0 - soft_urgency * (1.0 - tight_slack_mask))  # minor tie-breaker for borderline urgency
    )
    
    # Ensure finiteness and boundedness
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    return score
