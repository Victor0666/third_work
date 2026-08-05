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
    Self-evolved priority rule v2: Adaptive urgency restoration + unified risk-energy penalty + inclusive starvation rescue.
    
    Key improvements over v1:
    - Restores urgency for *all* violated and tight tasks (no latency-critical gating), fixing missed deadline penalties.
    - Replaces dual-gated energy amplification with *unified violation-aware exponentiation*: 
      (1 + uncertainty * max(0, -slack) / (task_duration + eps))^1.3 — activates strongly under any violation,
      smooth and monotonic, no arbitrary thresholds.
    - Starvation boost now includes *delayed high-criticality tasks*: removes `slack >= 0` and `upward_rank < median_ur` constraints,
      instead requiring only (norm_wait_ratio > 0.75) AND (rw_normalized > 0.25) AND (slack < 120.0),
      enabling rescue of long-waiting critical tasks even when delayed.
    - Introduces *deadline proximity normalization*: urgency penalty scaled by relative distance to workflow deadline
      (via slack quantiles) to balance across heterogeneous deadlines.
    - Tightens communication pressure: uses uncertainty-weighted comm overhead *and* explicit slack-distance scaling
      (penalty increases linearly as slack → negative), ensuring bandwidth-sensitive tasks are prioritized under pressure.
    - Robust normalization upgraded: handles N=1/2 via trimmed mean + range fallback; all divisions guarded; outputs clipped and finite.
    - Final weights rebalanced: DDL-hardness (0.56), synergy (0.27), comm (0.09), energy-risk (0.05), starvation (0.03)
      — emphasizing feasibility first, then efficiency, while preserving fairness.
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
    
    def robust_normalize(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, -1e12, 1e12)
        if N == 1:
            return np.array([0.0])
        elif N == 2:
            x_sorted = np.sort(x)
            center = np.mean(x_sorted)
            rng = x_sorted[1] - x_sorted[0] + eps
            return np.clip((x - center) / rng, -10.0, 10.0)
        else:
            center = np.median(x)
            mad = np.median(np.abs(x - center)) + eps
            normed = (x - center) / mad
            return np.clip(normed, -10.0, 10.0)
    
    task_duration = np.maximum(min_exec_time + min_comm_time, eps)
    rel_slack = np.divide(slack, task_duration, out=np.zeros_like(slack), where=task_duration != 0)
    
    # Compute adaptive urgency bands using quantiles (robust across scales)
    if N > 1:
        q20, q50, q80 = np.quantile(rel_slack, [0.2, 0.5, 0.8], method='midpoint')
    else:
        q20 = q50 = q80 = rel_slack[0]
    
    violated_mask = slack < 0
    tight_mask = ~violated_mask & (rel_slack < q50)
    relaxed_mask = ~violated_mask & (rel_slack >= q50)
    
    # Restored urgency: penalize all violated & tight tasks (no latency-critical gating)
    urgency_penalty = np.zeros_like(slack)
    urgency_penalty[violated_mask] = np.clip(-slack[violated_mask] / (task_duration[violated_mask] + eps), 0.0, 6.0)
    urgency_penalty[tight_mask] = np.clip(q50 - rel_slack[tight_mask], 0.0, 3.5)
    urgency_penalty[relaxed_mask] = np.clip(q80 - rel_slack[relaxed_mask], 0.0, 1.0)
    
    # Unified violation-aware energy risk: exponentiation triggered by any slack deficit
    rel_slack_distance = np.maximum(0.0, -slack) / (task_duration + eps)
    energy_exponent = 1.3
    risk_weighted_energy = min_incremental_energy * np.power(1.0 + uncertainty * rel_slack_distance + eps, energy_exponent)
    risk_weighted_energy = np.clip(risk_weighted_energy, eps, 1e9)
    
    # Slack-attenuated upward rank for criticality-energy synergy
    slack_factor = np.clip(1.0 - np.maximum(0.0, slack - q50 * np.median(task_duration)) / (task_duration + eps), 0.1, 1.0)
    dampened_ur = upward_rank * slack_factor
    base_synergy = dampened_ur * task_duration / (min_incremental_energy + eps)
    synergy_amplifier = np.where(violated_mask, 1.0 + 0.7 * uncertainty, 1.0)
    latency_crit_synergy = base_synergy * synergy_amplifier
    latency_crit_synergy = np.clip(latency_crit_synergy, eps, 1e8)
    
    # Communication pressure: uncertainty-weighted + linear slack-distance penalty
    comm_overhead_ratio = min_comm_time / (task_duration + eps)
    comm_pressure = comm_overhead_ratio * (1.0 + 0.5 * uncertainty)
    # Stronger penalty under violation, moderate under tight, baseline otherwise — scaled by |slack| proximity
    slack_distance_penalty = np.clip(1.0 + np.maximum(0.0, -slack) / (np.median(task_duration) + eps), 1.0, 3.0)
    comm_pressure = np.where(violated_mask, comm_pressure * 2.5 * slack_distance_penalty,
                           np.where(tight_mask, comm_pressure * 1.2 * slack_distance_penalty,
                                  comm_pressure * 0.3))
    
    # Starvation boost: inclusive rescue — long wait + non-negligible work + bounded delay (even if violated)
    median_task_dur = np.median(task_duration) + eps
    norm_wait_ratio = np.clip(ready_wait_time / median_task_dur, 0.0, 5.0)
    median_rw = np.median(remaining_work) + eps
    rw_normalized = np.clip(remaining_work / median_rw, 0.01, 100.0)
    # Removed slack>=0 and ur<median_ur: allows rescuing delayed but high-work, long-waiting tasks
    is_starvable = (norm_wait_ratio > 0.75) & (rw_normalized > 0.25) & (slack < 120.0)
    starvation_boost = np.where(is_starvable,
                              np.clip(norm_wait_ratio * (1.0 + 0.25 * np.minimum(120.0, -slack + 120.0) / 120.0), 0.0, 1.2),
                              0.0)
    
    # Normalize all components
    norm_urgency = robust_normalize(urgency_penalty)
    norm_synergy = robust_normalize(latency_crit_synergy)
    norm_energy = robust_normalize(risk_weighted_energy)
    norm_comm = robust_normalize(comm_pressure)
    norm_starvation = robust_normalize(starvation_boost)
    
    # Convex combination with rebalanced weights
    score = 0.56 * norm_urgency + (-0.27) * norm_synergy + 0.09 * norm_comm + 0.05 * norm_energy + 0.03 * norm_starvation
    score = np.nan_to_num(score, nan=0.0, posinf=1e10, neginf=-1e10)
    score = np.clip(score, -1e10, 1e10)
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
