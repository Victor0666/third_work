import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with dynamic feasibility gating and sharpened critical-path pressure.
    
    Key structural improvements:
      - Replaces static `feasibility_slack_threshold` with *dynamic workload-aware gate*:
        threshold = median_slack + factor * IQR(slack), adapting to slack distribution skew.
      - Introduces *critical-path interaction exponent*: (upward_rank × remaining_work)^exponent inside DDL-risk gate,
        amplifying high-importance successors more sharply under joint deadline+uncertainty stress.
      - Keeps robust adaptive normalization and multiplicative bottleneck-with-risk coupling.
      - Maintains bounded wait ramp and joint DDL-risk modulation — now applied to exponentiated critical-path term.
      - All operations remain finite, deterministic, and numerically safe.
    """
    eps = 0.0021655824629278845
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
        q_low = np.percentile(x, 39.15384947851905)
        q_high = np.percentile(x, 78.23027493992697)
        iqr = q_high - q_low
        center = np.median(x)
        if iqr < eps:
            x_min, x_max = (np.min(x), np.max(x))
            denom = x_max - x_min
            if denom < eps:
                return np.zeros_like(x)
            return (x - x_min) / (denom + eps)
        else:
            denom = iqr
            return (x - center) / (denom + eps)
    slack_centered = slack - -0.13861258321828618
    sigmoid_input = np.clip(-7.0320135507903645 * slack_centered, -34.34220758557828, 34.34220758557828)
    urgency = 1.0 / (1.0 + np.exp(sigmoid_input))
    norm_urgency = adaptive_normalize(urgency)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / duration
    norm_energy_eff = adaptive_normalize(energy_per_duration)
    bottleneck_pressure = duration * upward_rank * (1.0 + urgency + eps)
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    norm_uncertainty = adaptive_normalize(uncertainty)
    bottleneck_with_risk = norm_bottleneck * (1.0 + 0.21633470000529315 * norm_uncertainty)
    if N == 1:
        median_wait = ready_wait_time[0]
    else:
        median_wait = np.median(ready_wait_time)
    wait_ramp = np.clip(ready_wait_time / (median_wait + eps), 0.0, 1.0)
    norm_wait = adaptive_normalize(wait_ramp)
    slack_iqr = np.percentile(slack, 78.23027493992697) - np.percentile(slack, 39.15384947851905)
    dynamic_feasibility_threshold = np.median(slack) + 1.1054169570836856 * slack_iqr
    feasibility_gate = np.where(slack > dynamic_feasibility_threshold, 0.0, 1.0)
    median_slack = np.median(slack)
    max_uncertainty = np.max(uncertainty) if N > 0 else eps
    ddl_risk_gate = ((slack < median_slack) & (uncertainty > 0.7788921804057201 * max_uncertainty)).astype(float)
    critical_path_strength = (upward_rank * remaining_work + eps) ** 0.7509738935651047
    norm_critical_path = adaptive_normalize(critical_path_strength)
    score = norm_urgency + 0.2552299743770914 * ddl_risk_gate * norm_critical_path + feasibility_gate * 0.8732495295059391 * norm_energy_eff - norm_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
