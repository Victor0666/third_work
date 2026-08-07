import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: replaces sigmoid urgency with robust linear urgency gating,
    introduces host-load–aware energy normalization via multiplicative congestion scaling,
    and unifies risk coupling into a single adaptive urgency amplification term.
    
    Key structural improvements:
      - Replaces sigmoid urgency with bounded linear urgency gate: avoids parameter bloat and ensures monotonicity;
        maps slack → [-1,1] piecewise-linearly with explicit zero-crossing at slack=0.
      - Host-load–aware energy normalization: scales `min_incremental_energy` by `(1 + host_load_sensitivity * uncertainty)`
        before normalization — directly models energy inflation under resource contention without new inputs.
      - Unified risk coupling: urgency is multiplicatively amplified only when both slack < 0 AND uncertainty > threshold,
        replacing additive terms with cleaner, hierarchy-preserving gating.
      - Removes all redundant sigmoid parameters (`slack_sigmoid_*`, `sigmoid_clip_bound`) per reflection evidence.
      - Retains adaptive DDL-protection mask and arctan-saturated wait time; tightens IQR percentiles (22/78).
      - All operations remain finite, deterministic, and free of unbounded growth or state.
    """
    eps = 6.529980783863722e-05
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def iqr_normalize(x):
        x = np.copy(x)
        q_low = np.percentile(x, 22.009770274279532)
        q_high = np.percentile(x, 82.83975457820442)
        iqr = q_high - q_low
        center = np.median(x)
        denom = iqr if iqr > eps else np.max(np.abs(x - center)) + eps
        return (x - center) / (denom + eps)
    urgency_gate = np.clip(slack / (np.maximum(np.abs(np.median(slack)), eps) + eps), -1.0, 1.0)
    urgency_gate = -urgency_gate
    norm_urgency = iqr_normalize(urgency_gate)
    host_load_adjusted_energy = min_incremental_energy * (1.0 + 0.623791279385947 * uncertainty)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = host_load_adjusted_energy / duration
    norm_energy_eff = iqr_normalize(energy_per_duration)
    bottleneck_pressure = duration * (remaining_work + upward_rank + eps)
    norm_bottleneck = iqr_normalize(bottleneck_pressure)
    if N == 1:
        rank_percentile = np.array([1.0])
    else:
        sorted_ranks = np.sort(upward_rank)
        rank_idx = np.searchsorted(sorted_ranks, upward_rank, side='right')
        rank_percentile = rank_idx / (N + eps)
    critical_gate = np.where(rank_percentile >= 0.748910660574674, 1.0, 0.0)
    norm_rank = iqr_normalize(upward_rank)
    critical_bonus = critical_gate * norm_rank
    wait_scaled = ready_wait_time / (1.6265952180924057 + eps)
    wait_saturation = 2.0 / np.pi * np.arctan(wait_scaled)
    norm_wait = iqr_normalize(wait_saturation)
    median_slack = np.median(slack)
    max_uncertainty = np.max(uncertainty) if N > 0 else 1.0
    ddl_protection_mask = ((slack < median_slack) & (uncertainty > 0.6981319744388126 * max_uncertainty)).astype(float)
    norm_uncertainty = iqr_normalize(uncertainty)
    risk_amplification_mask = ((slack < 0) & (uncertainty > 0.6981319744388126 * max_uncertainty)).astype(float)
    risk_amplified_urgency = norm_urgency * (1.0 + 0.8817345442824007 * norm_uncertainty * risk_amplification_mask)
    score = risk_amplified_urgency + 0.2736379775681347 * norm_bottleneck + 1.0028422853038395 * norm_energy_eff - critical_bonus - norm_wait + ddl_protection_mask * norm_uncertainty
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
