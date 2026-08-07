import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule incorporating counterfactual evidence:
    - Removes inactive parameters (wait_saturation_time, critical_rank_percentile) per inactivity analysis
    - Replaces arctan wait saturation with bounded linear ramp + soft clamp to reduce complexity
    - Introduces multiplicative urgency × uncertainty coupling only under joint DDL pressure (tight slack AND high uncertainty)
    - Uses adaptive min-max fallback when IQR dispersion is low (more robust than pure IQR)
    - Eliminates critical-path bonus (no cross-scenario evidence supporting it) and replaces with bottleneck-coupled urgency
    - Prioritizes tasks whose execution unblocks successors *and* reduces deadline risk simultaneously
    """
    eps = 0.0020066931375522035
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
        q_low = np.percentile(x, 36.18294116063814)
        q_high = np.percentile(x, 77.26738273252171)
        iqr = q_high - q_low
        center = np.median(x)
        x_min, x_max = (np.min(x), np.max(x))
        range_val = x_max - x_min
        if iqr < eps * (range_val + eps):
            denom = range_val + eps
            return (x - x_min) / denom
        else:
            denom = iqr + eps
            return (x - center) / denom
    slack_centered = slack - 0.8677627648210304
    sigmoid_input = np.clip(-4.9583777022160405 * slack_centered, -22.14308947046867, 22.14308947046867)
    urgency_gate = 2.0 / (1.0 + np.exp(sigmoid_input)) - 1.0
    norm_urgency = adaptive_normalize(urgency_gate)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / duration
    norm_energy_eff = adaptive_normalize(energy_per_duration)
    bottleneck_pressure = duration * upward_rank * (1.0 + np.clip(urgency_gate, 0.0, 1.0) + eps)
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    median_slack = np.median(slack) if N > 0 else 0.0
    max_uncertainty = np.max(uncertainty) if N > 0 else 1.0
    ddl_protection_active = ((slack < median_slack) & (uncertainty > 0.7033267834689118 * max_uncertainty)).astype(float)
    urgency_uncertainty_coupling = ddl_protection_active * urgency_gate * uncertainty
    norm_coupling = adaptive_normalize(urgency_uncertainty_coupling)
    wait_normalized = np.clip(ready_wait_time / np.max(ready_wait_time + eps), 0.0, 1.0)
    norm_wait = adaptive_normalize(wait_normalized)
    score = norm_urgency + norm_coupling + 1.002866178783299 * norm_bottleneck + 1.5347003778091448 * norm_energy_eff - norm_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
