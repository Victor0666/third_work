import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
    - Adaptive IQR scaling: IQR width scaled by sqrt(N) via tunable factor to stabilize normalization for small ready sets.
    - Urgency fused with anti-starvation: ready_wait_time linearly added to slack before tanh mapping, eliminating separate boosted_wait term.
    - Tightened DDL-risk gate: activated only when (slack < median_slack AND uncertainty > local_median_uncertainty), removing global max dependency.
    - Critical bonus now applied multiplicatively to urgency (not additively to score), preserving urgency dominance while amplifying critical tasks under stress.
    - All operations guarded against zero/Nan/inf using eps and np.nan_to_num; no unbounded logic or side effects.
    """
    eps = 1e-06
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def adaptive_normalize(x):
        x = np.copy(x)
        q_low = np.percentile(x, 28.699645724730615)
        q_high = np.percentile(x, 77.96960131974862)
        iqr = q_high - q_low + eps
        n_scale = np.sqrt(max(N, 1.0)) * 1.3446806276609167
        center = np.median(x)
        return (x - center) / (iqr * n_scale + eps)
    fused_slack = slack + 0.5677272038219516 * ready_wait_time
    abs_fused_slack = np.abs(fused_slack)
    scale = np.median(abs_fused_slack) + eps
    tanh_urgency = np.tanh(-fused_slack / scale)
    urgency = np.clip((tanh_urgency + 1.0) / 2.0, 0.0, 1.0)
    norm_urgency = adaptive_normalize(urgency)
    norm_rank = adaptive_normalize(upward_rank)
    if N == 1:
        rank_percentile = np.array([1.0])
    else:
        sorted_norm_ranks = np.sort(norm_rank)
        rank_idx = np.searchsorted(sorted_norm_ranks, norm_rank, side='right')
        rank_percentile = rank_idx / (N + eps)
    critical_gate = np.where(rank_percentile >= 0.7711137230563908, 1.0, 0.0)
    amplified_urgency = urgency * (1.0 + critical_gate * norm_rank)
    norm_amplified_urgency = adaptive_normalize(amplified_urgency)
    duration = min_exec_time + min_comm_time
    coupled_energy = min_incremental_energy * (1.0 + 1.4519850556320844 * uncertainty + eps)
    bottleneck_pressure = duration * upward_rank * (1.0 + urgency) * (1.0 + uncertainty) * (coupled_energy + eps)
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    median_slack = np.median(slack)
    median_uncertainty = np.median(uncertainty) if N > 0 else eps
    ddl_risk_gate = ((slack < median_slack) & (uncertainty > 0.8233493111334625 * median_uncertainty)).astype(float)
    ddl_risk_amplification = ddl_risk_gate * norm_rank
    score = norm_amplified_urgency + 0.6571524031569468 * norm_bottleneck + ddl_risk_amplification
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
