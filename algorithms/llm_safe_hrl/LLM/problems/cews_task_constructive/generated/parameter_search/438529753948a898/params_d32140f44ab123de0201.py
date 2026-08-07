import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: bounded piecewise-linear urgency gate + restored binary DDL-protection.
    
    Key features:
      - Urgency is zero for positive slack, linear ramp for negative slack: 
        `max(-2, min(2, slope * (slack - offset)))` — fully bounded, no overflow.
      - DDL-protection gate is binary and activates only when slack < median AND 
        uncertainty > threshold * max_uncertainty; amplifies normalized uncertainty only then.
      - All parameters declared in schema are used; no unused or missing references.
      - No numeric literals beyond -2, -1, 0, 1, 2; all thresholds/weights are tunable parameters.
      - Uses np.finfo for safe finite clamping.
    """
    eps = 1.3510271544249893e-05
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
        q_low = np.percentile(x, 15.112300169662198)
        q_high = np.percentile(x, 85.03145526408753)
        iqr = q_high - q_low
        center = np.median(x)
        denom = iqr if iqr > eps else np.max(np.abs(x - center)) + eps
        return (x - center) / (denom + eps)
    urgency_raw = np.where(slack > 0, 0.0, 0.13466699098277413 * (slack - -0.9448203338185626))
    urgency_gate = np.clip(urgency_raw, -2.0, 2.0)
    norm_urgency = iqr_normalize(urgency_gate)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / duration
    norm_energy_eff = iqr_normalize(energy_per_duration)
    bottleneck_pressure = duration * (remaining_work + upward_rank + eps)
    norm_bottleneck = iqr_normalize(bottleneck_pressure)
    if N == 1:
        rank_percentile = np.array([1.0])
    else:
        sorted_ranks = np.sort(upward_rank)
        rank_idx = np.searchsorted(sorted_ranks, upward_rank, side='right')
        rank_percentile = rank_idx / (N + eps)
    critical_gate = np.where(rank_percentile >= 0.5654399548554797, 1.0, 0.0)
    norm_rank = iqr_normalize(upward_rank)
    critical_bonus = critical_gate * norm_rank
    wait_scaled = ready_wait_time / (11.81509016108761 + eps)
    wait_saturation = 2.0 / np.pi * np.arctan(wait_scaled)
    norm_wait = iqr_normalize(wait_saturation)
    median_slack = np.median(slack)
    max_uncertainty = np.max(uncertainty) if N > 0 else 1.0
    ddl_protection_active = ((slack < median_slack) & (uncertainty > 0.7981984866210996 * max_uncertainty)).astype(float)
    norm_uncertainty = iqr_normalize(uncertainty)
    ddl_risk_amplification = ddl_protection_active * norm_uncertainty
    score = norm_urgency + ddl_risk_amplification + 0.5168760590236532 * norm_bottleneck + 0.3749731212772419 * norm_energy_eff - critical_bonus - norm_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
