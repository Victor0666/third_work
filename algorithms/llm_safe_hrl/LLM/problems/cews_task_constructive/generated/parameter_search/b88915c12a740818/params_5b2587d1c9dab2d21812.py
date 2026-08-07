import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule incorporating:
    - Adaptive linear ramp for anti-starvation: saturation point = median_wait * factor (more robust than fixed time)
    - Slack-conditioned DDL-risk gate weighted by upward_rank to prioritize critical-path release under hard violation risk
    - Unified bottleneck pressure now includes remaining_work to reflect computational load downstream
    - Removed redundant risk-coupled term; replaced with direct upward_rank-weighted DDL-risk amplification
    - All normalizations use consistent adaptive IQR→min-max fallback; no unbounded ops or hidden state.
    """
    eps = 0.000119826464751383
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
        q_low = np.percentile(x, 38.17509895590506)
        q_high = np.percentile(x, 80.83679754111809)
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
    slack_centered = slack - 0.6257306622591112
    sigmoid_input = np.clip(-4.619788685688364 * slack_centered, -13.656015093390325, 13.656015093390325)
    urgency = 1.0 / (1.0 + np.exp(sigmoid_input))
    norm_urgency = adaptive_normalize(urgency)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / duration
    norm_energy_eff = adaptive_normalize(energy_per_duration)
    bottleneck_pressure = duration * upward_rank * (1.0 + urgency + eps) * (1.0 + uncertainty + eps) * (remaining_work + eps)
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    if N == 1:
        median_wait = ready_wait_time[0]
    else:
        median_wait = np.median(ready_wait_time)
    saturation_wait = median_wait * 1.613926010524312
    wait_ramp = np.clip(ready_wait_time / (saturation_wait + eps), 0.0, 1.0)
    norm_wait = adaptive_normalize(wait_ramp)
    max_uncertainty = np.max(uncertainty) if N > 0 else eps
    ddl_risk_condition = ((slack < 0.0) & (uncertainty > 0.599663072072125 * max_uncertainty)).astype(float)
    ddl_risk_amplification = ddl_risk_condition * upward_rank
    score = norm_urgency + 0.15698643006544633 * norm_bottleneck + ddl_risk_amplification + 1.8745925620127373 * norm_energy_eff - norm_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
