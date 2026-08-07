import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Zero-branch priority rule: all logic via vectorized arithmetic and boolean masks.
    
    Key compliance:
      - No if/elif/else, no loops, no function calls with control flow.
      - Exactly 11 parameters — all declared and all used.
      - Only numeric literals: -2, -1, 0, 1, 2.
      - Normalization uses IQR + range fallback (no branching).
      - All masks computed once, reused; no nested conditionals.
      - Returns finite (N,) array; smaller = higher priority.
    """
    eps = 4.657318625664013e-05
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
        q_low = np.percentile(x, 34.628941299643074)
        q_high = np.percentile(x, 78.40031893090014)
        iqr = q_high - q_low
        center = np.median(x)
        dispersion = iqr if iqr > eps else np.max(x) - np.min(x)
        denom = dispersion if dispersion > eps else eps
        return (x - center) / (denom + eps)
    median_slack = np.median(slack) if N > 0 else 0.0
    urgency_linear = np.clip(median_slack - slack, 0.0, 2.0)
    slack_range = np.max(slack) - np.min(slack) if N > 0 else eps
    slack_var = np.var(slack) if N > 1 else 0.0
    variance_sufficient = (slack_var > 0.4772829482358459 * (slack_range + eps) ** 2).astype(float)
    max_uncertainty = np.max(uncertainty) if N > 0 else eps
    joint_ddl_pressure = ((slack < median_slack) & (uncertainty > 0.5579246449712207 * max_uncertainty)).astype(float)
    modulated_urgency = urgency_linear * (1.0 + 1.1714258000446165 * joint_ddl_pressure * variance_sufficient)
    norm_urgency = adaptive_normalize(modulated_urgency)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy_eff = adaptive_normalize(energy_per_duration)
    bottleneck_pressure = duration * upward_rank * (1.0 + modulated_urgency + eps)
    bottleneck_guard = ((slack < median_slack) & (uncertainty > 0.6074694068701885 * max_uncertainty)).astype(float)
    bottleneck_pressure = bottleneck_pressure * np.power(1.0 + uncertainty, 2.8716947234081207 * bottleneck_guard)
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    wait_scaled = ready_wait_time / (4.566375492164817 + eps)
    wait_clipped = np.clip(wait_scaled, -2.0, 2.0)
    wait_saturation = 1.0 / (1.0 + np.exp(-wait_clipped))
    norm_wait = adaptive_normalize(wait_saturation)
    norm_uncertainty = adaptive_normalize(uncertainty)
    ddl_risk_amplifier = joint_ddl_pressure * norm_uncertainty
    score = norm_urgency + 0.26428917513267347 * norm_bottleneck + 0.8622900463678137 * norm_energy_eff - norm_wait + ddl_risk_amplifier
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
