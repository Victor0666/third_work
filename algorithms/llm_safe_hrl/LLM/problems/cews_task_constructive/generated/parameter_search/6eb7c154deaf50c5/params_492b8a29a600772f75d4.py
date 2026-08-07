import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with robust fallback normalization and multiplicative DDL-protection gating.
    
    Key improvements over v1:
      - Replaces fragile IQR-only normalization with adaptive min-max fallback when dispersion is low
      - Enforces *multiplicative* DDL-protection: only amplifies uncertainty signal when both slack < 0 AND uncertainty exceeds threshold
      - Removes unstable convex penalty; retains pure sigmoid urgency for smooth, bounded gradient
      - Simplifies bottleneck coupling to avoid over-normalization cascades
      - Ensures all operations are numerically guarded and finite at every step
    """
    eps = 0.01495304030770467
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def robust_normalize(x):
        x = np.copy(x)
        q_low = np.percentile(x, 10.0)
        q_high = np.percentile(x, 60.17019400769173)
        iqr = q_high - q_low
        x_range = np.max(x) - np.min(x)
        use_minmax = iqr < 0.10751855828138045 * (x_range + eps)
        if use_minmax:
            x_min, x_max = (np.min(x), np.max(x))
            denom = x_max - x_min
            norm_x = (x - x_min) / (denom + eps)
        else:
            center = np.median(x)
            denom = iqr if iqr > eps else np.max(np.abs(x - center)) + eps
            norm_x = (x - center) / (denom + eps)
        return norm_x
    slack_centered = slack - -0.40547683400141765
    sigmoid_input = np.clip(-2.190510801649661 * slack_centered, -13.396609403588927, 13.396609403588927)
    urgency_gate = 2.0 / (1.0 + np.exp(sigmoid_input)) - 1.0
    norm_urgency = robust_normalize(urgency_gate)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / duration
    norm_energy_eff = robust_normalize(energy_per_duration)
    bottleneck_pressure = duration * (remaining_work + upward_rank + eps)
    norm_bottleneck = robust_normalize(bottleneck_pressure)
    if N == 1:
        rank_percentile = np.array([1.0])
    else:
        sorted_ranks = np.sort(upward_rank)
        rank_idx = np.searchsorted(sorted_ranks, upward_rank, side='right')
        rank_percentile = rank_idx / (N + eps)
    critical_gate = np.where(rank_percentile >= 0.6981737607514386, 1.0, 0.0)
    norm_rank = robust_normalize(upward_rank)
    critical_bonus = critical_gate * norm_rank
    wait_scaled = ready_wait_time / (21.024524741084647 + eps)
    wait_saturation = 2.0 / np.pi * np.arctan(wait_scaled)
    norm_wait = robust_normalize(wait_saturation)
    max_uncertainty = np.max(uncertainty) if N > 0 else 1.0
    ddl_protection_active = ((slack < 0.0) & (uncertainty > 0.7501958819958623 * max_uncertainty)).astype(float)
    norm_uncertainty = robust_normalize(uncertainty)
    ddl_protection_signal = ddl_protection_active * norm_uncertainty
    score = norm_urgency + 0.1816418215594979 * norm_bottleneck + 0.48811969321515547 * norm_energy_eff - critical_bonus - norm_wait + ddl_protection_signal
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
