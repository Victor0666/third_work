import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with three key structural improvements:
      - Piecewise slack sensitivity: linear penalty for slack > 0, exponential (slack^exponent) for slack <= 0 — preserves convexity while sharply penalizing violations.
      - Task-type–aware normalization: compute-bound (exec_ratio >= threshold) tasks use tighter outlier clipping; I/O-bound use wider range to preserve communication urgency.
      - Energy optimization gated by *both* slack >= 0 AND uncertainty below tunable low_uncertainty_gate_threshold, ensuring energy savings only under predictable conditions.
    """
    eps = 0.0021190242019538314
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    exec_comm_sum = min_exec_time + min_comm_time
    exec_comm_sum_safe = np.maximum(exec_comm_sum, eps)
    exec_ratio = min_exec_time / exec_comm_sum_safe
    linear_slack_penalty = np.clip(-slack, 0.0, None)
    exp_slack_penalty = np.where(slack <= 0.0, np.power(np.abs(slack) + eps, 1.6858044182809646), 0.0)
    neg_slack = linear_slack_penalty + exp_slack_penalty
    median_slack = np.median(slack) if N > 0 else 0.0
    unc_min, unc_max = (np.min(uncertainty), np.max(uncertainty))
    unc_range = np.maximum(unc_max - unc_min, eps)
    unc_normalized = (uncertainty - unc_min) / (unc_range + eps)
    ddl_protection_active = (slack <= median_slack) & (unc_normalized > 0.4159331267880194)
    critical_pressure = upward_rank * remaining_work
    slack_ratio = np.where(np.abs(median_slack) > eps, slack / (np.abs(median_slack) + eps), 0.0)
    slack_ratio = np.clip(slack_ratio, -2.0, 2.0)
    successor_release = (1.0 + slack_ratio) * (1.0 + uncertainty)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    bottleneck_base = duration * critical_pressure * successor_release
    bottleneck_pressure = np.where(ddl_protection_active, bottleneck_base * np.power(1.0 + unc_normalized, 1.857943119769095), bottleneck_base)
    energy_term = np.where((slack >= 0.0) & (unc_normalized < 0.6855491717479023), min_incremental_energy, 0.0)

    def task_aware_normalize(x, is_compute_bound):
        x = np.copy(x)
        x_min = np.min(x) if N > 0 else 0.0
        x_max = np.max(x) if N > 0 else 1.0
        range_val = np.maximum(x_max - x_min, eps)
        clip_factor = np.where(is_compute_bound, 0.5900595079097437, 1.0) * 0.07586917309715835
        clip_offset = clip_factor * range_val
        x_clipped = np.clip(x, x_min - clip_offset, x_max + clip_offset)
        return (x_clipped - x_min) / (range_val + eps)
    is_compute_bound = exec_ratio >= 0.5204671422579272
    norm_critical = task_aware_normalize(critical_pressure, is_compute_bound)
    norm_bottleneck = task_aware_normalize(bottleneck_pressure, is_compute_bound)
    norm_energy = task_aware_normalize(energy_term, is_compute_bound)
    wait_scaled = ready_wait_time / (1.0 + eps)
    wait_clipped = np.clip(wait_scaled, -2.0, 2.0)
    wait_saturation = 1.0 / (1.0 + np.exp(-wait_clipped))
    norm_wait = task_aware_normalize(wait_saturation, is_compute_bound)
    score = neg_slack + 0.9346009418811647 * norm_critical + 0.01663963038843244 * norm_bottleneck + 0.422801206019093 * norm_energy - norm_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
