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
    Self-evolved priority rule v3: Adaptive deadline-hardness with dynamic quantile anchoring,
    synergistic criticality-energy coupling via slack-distance-aware gating, and starvation-avoidance
    with *work-significance-weighted wait pressure* and *latency-elastic normalization*.
    
    Key advances:
      - Dynamic quantile anchoring: replaces fixed quantiles with *task-duration-scaled relative slack*
        and computes quantiles on clipped rel_slack ([-10, 10]) for numerical stability across extreme workloads.
      - Synergy refinement: uses *asymmetric slack-gating* — amplification only when slack < q30 (not median),
        enabling earlier intervention on high-risk tasks; multiplier now includes normalized uncertainty × ur_ratio.
      - Starvation boost upgraded to *work-significance-weighted wait pressure*: 
        (ready_wait_time / task_duration) × sigmoid(remaining_work / median_rw) × (1 − sigmoid(slack)),
        ensuring fairness scales with both computational weight and urgency deficit.
      - Energy risk scoring sharpened with *exponential violation penalty*: exp(0.5 * rel_slack_distance) instead of power law,
        yielding stronger penalty for severe violations while remaining smooth and bounded.
      - Latency-elastic normalization: replaces MAD with *quantile-based robust scaling* (IQR + eps) and clips outliers
        before normalization to avoid distortion from extreme values; deterministic fallback for N=1 remains.
      - Final weights rebalanced to emphasize urgency dominance (0.65), synergy efficiency (0.25), energy control (0.06),
        starvation relief (0.03), and communication overhead (0.01) — tightening DDL-hardness focus.
    '''
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
    
    def robust_normalize_quantile(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, -1e12, 1e12)
        if N == 1:
            return np.array([0.0])
        q25 = np.quantile(x, 0.25, method='midpoint')
        q75 = np.quantile(x, 0.75, method='midpoint')
        iqr = q75 - q25 + eps
        center = np.median(x)
        normed = (x - center) / iqr
        return np.clip(normed, -10.0, 10.0)
    
    task_duration = np.maximum(min_exec_time + min_comm_time, eps)
    rel_slack = np.divide(slack, task_duration, out=np.zeros_like(slack), where=task_duration != 0)
    # Clip for stable quantile computation under extreme ratios
    rel_slack_clipped = np.clip(rel_slack, -10.0, 10.0)
    
    if N > 1:
        q10, q30, q50, q70, q90 = np.quantile(rel_slack_clipped, [0.1, 0.3, 0.5, 0.7, 0.9], method='midpoint')
    else:
        q10 = q30 = q50 = q70 = q90 = rel_slack_clipped[0]
    
    violated_mask = slack < 0
    tight_mask = ~violated_mask & (rel_slack < q50)
    relaxed_mask = ~violated_mask & (rel_slack >= q50)
    
    # Urgency penalty: sharper violation response, smoother interpolation in tight regime
    urgency_penalty = np.zeros_like(slack)
    urgency_penalty[violated_mask] = np.clip(-slack[violated_mask] / (task_duration[violated_mask] + eps), 0.0, 10.0)
    urgency_penalty[tight_mask] = np.clip(q50 - rel_slack[tight_mask], 0.0, 4.5)
    urgency_penalty[relaxed_mask] = np.clip(q90 - rel_slack[relaxed_mask], 0.0, 1.0)
    
    # Slack factor for criticality damping — tighter threshold (q30) enables earlier prioritization of high-rank at risk
    slack_factor = np.clip(1.0 - np.maximum(0.0, slack - q30 * np.median(task_duration)) / (task_duration + eps), 0.02, 1.0)
    dampened_ur = upward_rank * slack_factor
    
    base_synergy = dampened_ur * task_duration / (min_incremental_energy + eps)
    
    # Asymmetric synergy amplification: only active when slack < q30 (early warning zone), not median
    median_ur = np.median(upward_rank) + eps
    ur_ratio = np.clip(upward_rank / median_ur, 0.1, 10.0)
    synergy_amplifier = np.where(slack < q30 * np.median(task_duration), 
                                1.0 + 0.9 * uncertainty * ur_ratio, 
                                1.0)
    latency_crit_synergy = base_synergy * synergy_amplifier
    latency_crit_synergy = np.clip(latency_crit_synergy, eps, 1e8)
    
    # Exponential energy penalty focused strictly on violation severity
    rel_slack_distance = np.maximum(0.0, -slack) / (task_duration + eps)
    risk_weighted_energy = min_incremental_energy * np.exp(0.5 * rel_slack_distance)
    risk_weighted_energy = np.clip(risk_weighted_energy, eps, 1e9)
    
    # Communication pressure: reduced weight, more selective scaling
    comm_overhead_ratio = min_comm_time / (task_duration + eps)
    comm_pressure = comm_overhead_ratio * (1.0 + 0.3 * uncertainty)
    comm_pressure = np.where(violated_mask, comm_pressure * 3.0, 
                            np.where(tight_mask, comm_pressure * 0.6, comm_pressure * 0.05))
    
    # Work-significance-weighted wait pressure: balances wait time, computational relevance, and urgency deficit
    median_task_dur = np.median(task_duration) + eps
    median_rw = np.median(remaining_work) + eps
    wait_ratio = np.clip(ready_wait_time / median_task_dur, 0.0, 5.0)
    rw_sigmoid = 1.0 / (1.0 + np.exp(-(remaining_work / median_rw - 0.5) / 0.2))
    slack_deficit_sigmoid = 1.0 / (1.0 + np.exp((slack - q30 * np.median(task_duration)) / (np.maximum(task_duration, 1.0) + eps)))
    wait_boost = wait_ratio * rw_sigmoid * slack_deficit_sigmoid
    wait_boost = np.clip(wait_boost, 0.0, 1.8)
    
    # Normalize components using robust quantile scaling
    norm_urgency = robust_normalize_quantile(urgency_penalty)
    norm_synergy = robust_normalize_quantile(latency_crit_synergy)
    norm_energy = robust_normalize_quantile(risk_weighted_energy)
    norm_comm = robust_normalize_quantile(comm_pressure)
    norm_wait = robust_normalize_quantile(wait_boost)
    
    # Final weighted score: urgency dominates; synergy strongly supports feasible critical path; others are secondary
    score = 0.65 * norm_urgency - 0.25 * norm_synergy + 0.06 * norm_energy + 0.01 * norm_comm + 0.03 * norm_wait
    score = np.nan_to_num(score, nan=0.0, posinf=1e10, neginf=-1e10)
    score = np.clip(score, -1e10, 1e10)
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
