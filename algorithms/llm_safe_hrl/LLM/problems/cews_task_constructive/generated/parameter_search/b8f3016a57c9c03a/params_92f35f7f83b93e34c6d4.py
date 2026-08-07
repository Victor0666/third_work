import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: replaces hard feasibility gating with soft exponential slack decay on energy,
    reintroduces conditional slack_sigmoid_offset under high uncertainty, and upgrades bottleneck coupling to
    additive uncertainty interaction (not multiplicative) for monotonic, well-bounded risk amplification.
    
    Key structural improvements:
      - Soft exponential decay on energy term: weight = exp(-decay_rate * max(0, slack)), smooth & differentiable
      - Conditional slack offset: applied only when uncertainty exceeds threshold, improving urgency calibration in noisy regimes
      - Additive uncertainty interaction: bottleneck_pressure + weight * norm_uncertainty * bottleneck_pressure,
        preserving sign and avoiding zero-crossing artifacts
      - Tunable IQR percentiles (22/78) further improve outlier robustness
      - All operations guarded; deterministic; finite output guaranteed
    """
    eps = 0.002909826980293831
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
        q_low = np.percentile(x, 13.499285940209594)
        q_high = np.percentile(x, 70.30867229307704)
        iqr = q_high - q_low
        center = np.median(x)
        denom = iqr if iqr > eps else np.max(np.abs(x - center)) + eps
        return (x - center) / (denom + eps)
    max_uncertainty = np.max(uncertainty) if N > 0 else 1.0
    high_uncertainty_mask = (uncertainty > 0.7077261464745699 * max_uncertainty).astype(float)
    effective_offset = high_uncertainty_mask * 0.07479569645506956
    slack_centered = slack - effective_offset
    sigmoid_input = np.clip(-5.852866970647192 * slack_centered, -35.00603890480483, 35.00603890480483)
    urgency_gate = 2.0 / (1.0 + np.exp(sigmoid_input)) - 1.0
    norm_urgency = iqr_normalize(urgency_gate)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / duration
    norm_energy_eff = iqr_normalize(energy_per_duration)
    bottleneck_pressure = duration * (remaining_work + upward_rank + eps)
    norm_bottleneck = iqr_normalize(bottleneck_pressure)
    norm_uncertainty = iqr_normalize(uncertainty)
    bottleneck_with_risk = norm_bottleneck + 0.015379150158583458 * norm_uncertainty * norm_bottleneck
    if N == 1:
        rank_percentile = np.array([1.0])
    else:
        sorted_ranks = np.sort(upward_rank)
        rank_idx = np.searchsorted(sorted_ranks, upward_rank, side='right')
        rank_percentile = rank_idx / (N + eps)
    critical_gate = np.where(rank_percentile >= 0.6508844970933692, 1.0, 0.0)
    norm_rank = iqr_normalize(upward_rank)
    critical_bonus = critical_gate * norm_rank
    wait_scaled = ready_wait_time / (13.564112843895828 + eps)
    wait_saturation = 2.0 / np.pi * np.arctan(wait_scaled)
    norm_wait = iqr_normalize(wait_saturation)
    slack_positive = np.maximum(slack, 0.0)
    energy_weight = np.exp(-0.015379150158583458 * slack_positive)
    median_slack = np.median(slack)
    ddl_protection_active = ((slack < median_slack) & (uncertainty > 0.7077261464745699 * max_uncertainty)).astype(float)
    score = norm_urgency + 0.7147687424582343 * bottleneck_with_risk + energy_weight * 1.2033032505241676 * norm_energy_eff - critical_bonus - norm_wait + ddl_protection_active * norm_uncertainty
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
