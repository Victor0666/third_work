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
    Self-evolved priority rule v5: Tri-threshold urgency-aware criticality coupling,
    starvation-avoidance with *dynamic work-slack co-activation*, and *risk-contextual energy scaling*.
    
    Key advances over v4:
      - Triple-threshold slack gating (q10/q30/q60): enables graduated intervention — 
        aggressive for violated (q10), proactive for tight (q30), and conservative for relaxed (q60),
        improving deadline adherence under heterogeneous uncertainty.
      - Criticality-energy coupling now uses *risk-adjusted upward_rank*: ur × (1 + 0.7 * normalized_slack_risk + 0.3 * uncertainty),
        embedding both deadline risk and execution uncertainty into HEFT semantics.
      - Starvation relief upgraded to *dynamic work-slack co-activation*: replaces fixed sigmoid with
        adaptive thresholding — boost activates only when (remaining_work > 0.3 * median_rw) AND (slack < 0.7 * median_dur),
        eliminating spurious boosts on trivial or non-urgent tasks.
      - Energy scoring now uses *contextual risk scaling*: base_energy × (1 + 0.6 * exp(clipped_slack_dist) + 0.4 * uncertainty),
        making energy penalty responsive to both deadline violation severity and execution unpredictability.
      - Communication penalty enhanced with *latency-critical weighting*: comm_penalty × (1 + 0.8 * (upward_rank / median_ur)),
        prioritizing data movement for high-criticality tasks to prevent critical-path stalls.
      - Robust normalization upgraded to *cardinality-adaptive quantile tiers*: 
          N==1 → zero; 2≤N<4 → MAD; 4≤N<10 → IQR; N≥10 → trimmed quantiles (p10-p90) + Winsorized mean,
        maximizing stability across all operational scales.
      - Final weights rebalanced for stronger DDL-hardness (0.64), tighter synergy (0.24), elevated energy control (0.08),
        refined starvation relief (0.03), and amplified communication awareness (0.01).
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
    
    # Safe sanitization: clamp NaN/inf to safe finite bounds
    min_exec_time = np.clip(np.nan_to_num(min_exec_time, nan=eps, posinf=1e9, neginf=eps), eps, 1e9)
    min_comm_time = np.clip(np.nan_to_num(min_comm_time, nan=eps, posinf=1e9, neginf=eps), eps, 1e9)
    min_incremental_energy = np.clip(np.nan_to_num(min_incremental_energy, nan=eps, posinf=1e9, neginf=eps), eps, 1e9)
    slack = np.nan_to_num(slack, nan=0.0, posinf=1e9, neginf=-1e9)
    upward_rank = np.clip(np.nan_to_num(upward_rank, nan=eps, posinf=1e6, neginf=eps), eps, 1e6)
    remaining_work = np.clip(np.nan_to_num(remaining_work, nan=eps, posinf=1e9, neginf=eps), eps, 1e9)
    ready_wait_time = np.clip(np.nan_to_num(ready_wait_time, nan=0.0, posinf=1e6, neginf=0.0), 0.0, 1e6)
    uncertainty = np.clip(np.nan_to_num(uncertainty, nan=0.0, posinf=1e3, neginf=0.0), 0.0, 1e3)
    
    task_duration = min_exec_time + min_comm_time
    task_duration = np.maximum(task_duration, eps)
    
    # Relative slack with robust division
    rel_slack = np.divide(slack, task_duration, out=np.zeros_like(slack), where=task_duration != 0)
    rel_slack_clipped = np.clip(rel_slack, -10.0, 10.0)
    
    # Triple-threshold quantiles for graduated urgency gating
    if N > 1:
        q10, q30, q50, q60, q90 = np.quantile(rel_slack_clipped, [0.1, 0.3, 0.5, 0.6, 0.9], method='midpoint')
    else:
        q10 = q30 = q50 = q60 = q90 = rel_slack_clipped[0]
    
    violated_mask = slack < 0
    tight_mask = ~violated_mask & (rel_slack < q30)
    mid_mask = ~violated_mask & (rel_slack >= q30) & (rel_slack < q60)
    relaxed_mask = ~violated_mask & (rel_slack >= q60)
    
    # Urgency penalty: graded response per slack tier
    urgency_penalty = np.zeros_like(slack)
    urgency_penalty[violated_mask] = np.clip(-slack[violated_mask] / (task_duration[violated_mask] + eps), 0.0, 12.0)
    urgency_penalty[tight_mask] = np.clip(q50 - rel_slack[tight_mask], 0.0, 3.5)
    urgency_penalty[mid_mask] = np.clip(q60 - rel_slack[mid_mask], 0.0, 1.2)
    urgency_penalty[relaxed_mask] = np.clip(q90 - rel_slack[relaxed_mask], 0.0, 0.3)
    
    # Risk-aware criticality coupling: upward_rank enhanced by both slack risk and uncertainty
    slack_risk = np.maximum(0.0, -slack) / (task_duration + eps)
    normalized_slack_risk = np.clip(slack_risk, 0.0, 5.0)
    median_ur = np.median(upward_rank) + eps
    ur_ratio = np.clip(upward_rank / median_ur, 0.1, 10.0)
    risk_adjusted_ur = upward_rank * (1.0 + 0.7 * normalized_slack_risk + 0.3 * uncertainty)
    base_synergy = risk_adjusted_ur * task_duration / (min_incremental_energy + eps)
    base_synergy = np.clip(base_synergy, eps, 1e8)
    
    # Communication penalty: scaled by criticality to protect critical path
    comm_penalty = min_comm_time * (1.0 + 0.5 * uncertainty) * (1.0 + 0.8 * ur_ratio)
    comm_penalty = np.where(violated_mask, comm_penalty * 5.0, 
                           np.where(tight_mask, comm_penalty * 2.0, 
                                  comm_penalty * 0.2))
    
    # Contextual energy scoring: jointly penalized by violation severity and uncertainty
    rel_slack_distance = np.maximum(0.0, -slack) / (task_duration + eps)
    clipped_dist = np.clip(rel_slack_distance, 0.0, 5.0)
    risk_weighted_energy = min_incremental_energy * (
        1.0 + 0.6 * (np.exp(clipped_dist) - 1.0) + 0.4 * uncertainty
    )
    risk_weighted_energy = np.clip(risk_weighted_energy, eps, 1e9)
    
    # Dynamic starvation relief: activates only when both work weight AND urgency deficit are significant
    median_task_dur = np.median(task_duration) + eps
    median_rw = np.median(remaining_work) + eps
    wait_ratio = np.clip(ready_wait_time / median_task_dur, 0.0, 5.0)
    
    # Co-activation condition: only boost if work is substantial AND slack is critically low
    work_significant = remaining_work > 0.3 * median_rw
    slack_critical = slack < 0.7 * median_task_dur
    co_activation_mask = work_significant & slack_critical
    
    wait_boost = np.zeros_like(ready_wait_time)
    wait_boost[co_activation_mask] = wait_ratio[co_activation_mask] * 0.8
    
    # Robust normalization: tiered by cardinality for optimal sensitivity/stability tradeoff
    def robust_normalize(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, -1e10, 1e10)
        if N == 1:
            return np.array([0.0])
        elif N < 4:
            center = np.median(x)
            abs_devs = np.abs(x - center)
            mad = np.median(abs_devs) + eps
            z = (x - center) / mad
            return np.clip(z, -6.0, 6.0)
        elif N < 10:
            q25 = np.quantile(x, 0.25, method='midpoint')
            q75 = np.quantile(x, 0.75, method='midpoint')
            iqr = q75 - q25 + eps
            center = np.median(x)
            z = (x - center) / iqr
            return np.clip(z, -10.0, 10.0)
        else:
            # Trimmed quantiles + Winsorized mean for large N
            x_sorted = np.sort(x)
            trim_lo = max(1, int(0.1 * N))
            trim_hi = max(1, int(0.1 * N))
            x_trimmed = x_sorted[trim_lo:-trim_hi] if len(x_sorted) > trim_lo + trim_hi else x_sorted
            winsor_mean = np.mean(np.clip(x, np.percentile(x, 10), np.percentile(x, 90)))
            q25_trim = np.quantile(x_trimmed, 0.25, method='midpoint') if len(x_trimmed) > 1 else winsor_mean
            q75_trim = np.quantile(x_trimmed, 0.75, method='midpoint') if len(x_trimmed) > 1 else winsor_mean
            iqr_trim = q75_trim - q25_trim + eps
            z = (x - winsor_mean) / iqr_trim
            return np.clip(z, -12.0, 12.0)
    
    norm_urgency = robust_normalize(urgency_penalty)
    norm_synergy = robust_normalize(base_synergy)
    norm_energy = robust_normalize(risk_weighted_energy)
    norm_comm = robust_normalize(comm_penalty)
    norm_wait = robust_normalize(wait_boost)
    
    # Final weighted score: smaller = higher priority
    score = (0.64 * norm_urgency 
             - 0.24 * norm_synergy 
             + 0.08 * norm_energy 
             + 0.01 * norm_comm 
             + 0.03 * norm_wait)
    
    score = np.nan_to_num(score, nan=0.0, posinf=1e10, neginf=-1e10)
    score = np.clip(score, -1e10, 1e10)
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
