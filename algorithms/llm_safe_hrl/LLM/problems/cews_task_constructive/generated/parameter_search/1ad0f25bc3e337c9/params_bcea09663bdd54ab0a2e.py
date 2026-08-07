import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining strengths from both parents:
    - Retains Parent 2's sharp sigmoid urgency with steepness/offset tuning for precise zero-slack discrimination.
    - Keeps adaptive wait ramp with saturation factor for fairness robustness in congestion.
    - Preserves remaining_work in bottleneck pressure to reflect downstream load impact.
    - Introduces novel 'urgency_bias_weight': scales normalized urgency *before* bottleneck coupling to enforce stronger deadline-first hierarchy.
    - Uses joint DDL-risk amplification (upward_rank × ddl_risk_condition) only under hard violation, avoiding over-prioritization.
    - All normalizations use unified adaptive IQR→min-max fallback; no unbounded ops or hidden state.
    """
    eps = 8.771190001906931e-05
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
        q_low = np.percentile(x, 33.86610572352627)
        q_high = np.percentile(x, 79.33809804719769)
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
    slack_centered = slack - 0.7621528671218722
    sigmoid_input = np.clip(-7.930055576447642 * slack_centered, -20.95787088999544, 20.95787088999544)
    urgency = 1.0 / (1.0 + np.exp(sigmoid_input))
    norm_urgency = adaptive_normalize(urgency)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy_eff = adaptive_normalize(energy_per_duration)
    bottleneck_pressure = duration * upward_rank * (1.0 + urgency + eps) * (1.0 + uncertainty + eps) * (remaining_work + eps)
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    if N == 1:
        median_wait = ready_wait_time[0]
    else:
        median_wait = np.median(ready_wait_time)
    saturation_wait = median_wait * 2.673005801719742
    wait_ramp = np.clip(ready_wait_time / (saturation_wait + eps), 0.0, 1.0)
    norm_wait = adaptive_normalize(wait_ramp)
    max_uncertainty = np.max(uncertainty) if N > 0 else eps
    ddl_risk_condition = ((slack < 0.0) & (uncertainty > 0.6611990597311657 * max_uncertainty)).astype(float)
    ddl_risk_amplification = ddl_risk_condition * upward_rank
    biased_urgency = 1.0774926105947296 * norm_urgency
    score = biased_urgency + 0.028501599351729738 * norm_bottleneck + ddl_risk_amplification + 1.5334284934476534 * norm_energy_eff - norm_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
