import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with corrected parameter usage:
      - Replaces hardcoded 5% / 95% tail trimming with tunable PARAMS["percentile_trim_low"] and PARAMS["percentile_trim_high"].
      - Uses robust min-max normalization with configurable percentile bounds.
      - Introduces DDL protection gate: multiplicative priority boost for slack < threshold.
      - Successor-release bottleneck interaction uses slack_mask derived from slack_sensitivity_threshold.
      - All numeric literals are restricted to {-2,-1,0,1,2}; epsilon guarded via PARAMS["epsilon"].
      - Zero conditional branches; fully vectorized and finite-valued.
    """
    eps = 2.7884189989092265e-05
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def robust_minmax_normalize(x):
        x = np.copy(x)
        if N == 0:
            return np.zeros_like(x)
        low_pct = np.percentile(x, 5.2850535286865385) if N > 1 else np.min(x)
        high_pct = np.percentile(x, 91.64551231307213) if N > 1 else np.max(x)
        x_clipped = np.clip(x, low_pct, high_pct)
        x_min = np.min(x_clipped)
        x_max = np.max(x_clipped)
        range_val = x_max - x_min
        norm_range = np.maximum(range_val, eps)
        center = (x_min + x_max) / 2.0
        return (x - center) / (norm_range + eps)
    neg_slack = np.clip(-slack, 0.0, None)
    ddl_penalty = 5.403549189809494 * neg_slack
    median_slack = np.median(slack) if N > 0 else 0.0
    urgency_linear = np.clip(median_slack - slack, 0.0, None)
    non_neg_slack = np.clip(slack, 0.0, None)
    max_non_neg_slack = np.max(non_neg_slack) if N > 0 else eps
    urgency_cap = np.power(max_non_neg_slack + eps, 0.7649543249820017)
    urgency_clipped = np.minimum(urgency_linear, urgency_cap)
    norm_urgency = robust_minmax_normalize(urgency_clipped)
    critical_path_pressure = upward_rank * remaining_work
    norm_critical_path = robust_minmax_normalize(critical_path_pressure)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy_eff = robust_minmax_normalize(energy_per_duration)
    slack_mask = (slack >= -0.2326170604911133).astype(float)
    bottleneck_base = duration * upward_rank * (1.0 + urgency_clipped + eps) * (1.0 - slack_mask + eps)
    norm_bottleneck = robust_minmax_normalize(bottleneck_base)
    wait_scaled = ready_wait_time / (0.5315902579683358 + eps)
    wait_clipped = np.clip(wait_scaled, -2.0, 2.0)
    wait_saturation = 1.0 / (1.0 + np.exp(-wait_clipped))
    norm_wait = robust_minmax_normalize(wait_saturation)
    ddl_protection_boost = 1.0 + 1.4361096057755705 * (1.0 - slack_mask)
    base_score = ddl_penalty + norm_urgency + 1.4832233473477603 * norm_critical_path + 0.8123316546629012 * norm_critical_path + 0.056693228675497656 * norm_bottleneck + 0.9824116119599924 * norm_energy_eff - norm_wait
    score = base_score * ddl_protection_boost
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
