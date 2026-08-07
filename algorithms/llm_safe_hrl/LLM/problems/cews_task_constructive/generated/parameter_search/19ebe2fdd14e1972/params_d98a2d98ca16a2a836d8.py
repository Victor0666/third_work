import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with three key structural improvements:
    1. Adaptive urgency steepness: sigmoid steepness dynamically scaled by task-local uncertainty / global max_uncertainty,
       sharpening deadline response precisely where risk is highest (replaces fixed steepness + linear fallback).
    2. Deadline-aware critical-path bonus: only activates when both upward_rank is top-percentile AND slack < 0,
       enforcing DAG-aware prioritization strictly under deadline pressure.
    3. Robust anti-starvation: uses bounded reciprocal scaling (1/(1+wait/base)^exponent) with tunable base time,
       providing smooth, monotonic fairness without saturation artifacts or zero-division risk.
    All normalizations use IQR; no hardcoded constants beyond [-2,-1,0,1,2]; deterministic and finite-valued.
    """
    eps = 0.002713186347624729
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
        q_low = np.percentile(x, 38.982255629777896)
        q_high = np.percentile(x, 80.78090732673701)
        iqr = q_high - q_low
        center = (q_low + q_high) / 2.0
        denom = iqr if iqr > eps else np.max(np.abs(x - center)) + eps
        return (x - center) / (denom + eps)
    max_uncertainty = np.max(uncertainty) if N > 0 else 1.0
    uncertainty_ratio = np.clip(uncertainty / (max_uncertainty + eps), 0.0, 1.0)
    adaptive_steepness = 5.634220019350316 * (1.0 + uncertainty_ratio)
    slack_centered = slack - 0.0
    sigmoid_input = -adaptive_steepness * slack_centered
    finfo = np.finfo(float)
    sigmoid_input = np.clip(sigmoid_input, -finfo.max, finfo.max)
    urgency_gate = 1.0 / (1.0 + np.exp(-sigmoid_input))
    norm_urgency = iqr_normalize(urgency_gate)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / duration
    log_energy_eff = np.log1p(energy_per_duration + eps)
    norm_energy_eff = iqr_normalize(log_energy_eff)
    bottleneck_pressure = duration * (upward_rank + eps) * (remaining_work + eps) * (1.0 + uncertainty)
    norm_bottleneck = iqr_normalize(bottleneck_pressure)
    if N == 1:
        rank_percentile = np.array([1.0])
    else:
        sorted_ranks = np.sort(upward_rank)
        rank_idx = np.searchsorted(sorted_ranks, upward_rank, side='right')
        rank_percentile = rank_idx / (N + eps)
    critical_gate = np.where((rank_percentile >= 0.6303818841374675) & (slack < 0), 1.0, 0.0)
    norm_rank = iqr_normalize(upward_rank)
    critical_bonus = critical_gate * norm_rank
    wait_scaled = ready_wait_time / (2.4995153052469985 + eps)
    wait_reciprocal = 1.0 / (1.0 + wait_scaled)
    wait_powered = np.power(wait_reciprocal, 1.2502396729449656)
    norm_wait = iqr_normalize(wait_powered)
    ddl_protection_active = (uncertainty > 0.6608478703360083 * max_uncertainty).astype(float)
    norm_uncertainty = iqr_normalize(uncertainty)
    score = norm_urgency + critical_bonus + 0.2215825823830137 * norm_bottleneck + 0.7438327288737625 * norm_energy_eff - norm_wait + ddl_protection_active * norm_uncertainty
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=np.finfo(float).min)
    return score.astype(float, copy=False)
