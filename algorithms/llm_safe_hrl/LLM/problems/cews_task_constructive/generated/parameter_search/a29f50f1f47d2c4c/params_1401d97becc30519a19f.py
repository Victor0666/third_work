import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with:
    - Simplified urgency: direct sigmoid on slack (no offset/steepness parameters) → removes inactive params
    - Unified risk-aware bottleneck: (duration + comm_time) * upward_rank * (1 + sigmoid(-slack)) * (1 + uncertainty)
    - Anti-starvation via normalized wait time (no saturation ramp → simpler, more robust)
    - DDL-protection gate now uses joint condition: tight slack AND high uncertainty, activated only when both hold
    - Removed redundant terms (wait_ramp_saturation_factor, slack_sigmoid_* etc.) per inactivity evidence
    - All normalizations use adaptive IQR; fallback to min-max if dispersion too low.
    """
    eps = 0.000133655011664548
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
        q_low = np.percentile(x, 39.98783509581192)
        q_high = np.percentile(x, 89.74429463033745)
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
    sigmoid_input = np.clip(-slack, -19.45217059928582, 19.45217059928582)
    urgency = 1.0 / (1.0 + np.exp(sigmoid_input))
    norm_urgency = adaptive_normalize(urgency)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy_eff = adaptive_normalize(energy_per_duration)
    bottleneck_pressure = duration * upward_rank * (1.0 + urgency) * (1.0 + uncertainty + eps)
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    norm_wait = adaptive_normalize(ready_wait_time)
    max_uncertainty = np.max(uncertainty) if N > 0 else eps
    ddl_gate = ((slack < 0.0) & (uncertainty > 0.5664495904768413 * max_uncertainty)).astype(float)
    score = norm_urgency + 0.1302303964385939 * norm_bottleneck + 1.4965921991187723 * norm_energy_eff - norm_wait + ddl_gate * upward_rank
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
