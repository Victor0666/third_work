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
    eps = 1.7499611041821873e-05
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
        q_low = np.percentile(x, 38.9056116858641)
        q_high = np.percentile(x, 81.59987204463086)
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
    slack_centered = slack - 0.999977624693414
    sigmoid_input = np.clip(-5.162922647748033 * slack_centered, -34.38360517786428, 34.38360517786428)
    urgency = 1.0 / (1.0 + np.exp(sigmoid_input))
    norm_urgency = adaptive_normalize(urgency)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / duration
    norm_energy_eff = adaptive_normalize(energy_per_duration)
    bottleneck_pressure = duration * upward_rank * (1.0 + urgency + eps)
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    norm_uncertainty = adaptive_normalize(uncertainty)
    bottleneck_with_risk = norm_bottleneck * (1.0 + 0.09473278434166557 * norm_uncertainty)
    if N == 1:
        median_wait = ready_wait_time[0]
    else:
        median_wait = np.median(ready_wait_time)
    wait_ramp = np.clip(ready_wait_time / (median_wait + eps), 0.0, 1.0)
    norm_wait = adaptive_normalize(wait_ramp)
    slack_iqr = np.percentile(slack, 81.59987204463086) - np.percentile(slack, 38.9056116858641)
    dynamic_feasibility_threshold = np.median(slack) + 0.3122593680459518 * slack_iqr
    feasibility_gate = np.where(slack > dynamic_feasibility_threshold, 0.0, 1.0)
    median_slack = np.median(slack)
    max_uncertainty = np.max(uncertainty) if N > 0 else eps
    ddl_risk_gate = ((slack < median_slack) & (uncertainty > 0.8186869646069259 * max_uncertainty)).astype(float)
    critical_path_strength = (upward_rank * remaining_work + eps) ** 1.326951003064385
    norm_critical_path = adaptive_normalize(critical_path_strength)
    score = norm_urgency + 0.016622985547767907 * ddl_risk_gate * norm_critical_path + feasibility_gate * 1.8322035858431427 * norm_energy_eff - norm_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
