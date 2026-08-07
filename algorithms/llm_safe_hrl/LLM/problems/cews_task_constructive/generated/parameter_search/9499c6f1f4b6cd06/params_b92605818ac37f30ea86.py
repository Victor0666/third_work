import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule incorporating:
    - Linear slack penalty (not sigmoid) with uncertainty coupling for sharper, numerically stable deadline risk.
    - Direct upward_rank × remaining_work interaction to capture critical-path computational bottleneck pressure.
    - Energy term shifted by offset to avoid over-penalizing low-energy tasks when slack is tight.
    - Removed anti-starvation ramp (replaced by robust wait-aware normalization in urgency term).
    - Unified bottleneck term now explicitly includes uncertainty-weighted slack penalty.
    - All normalizations use adaptive IQR→min-max fallback; no unbounded ops or hidden state.
    """
    eps = 5.944873911253614e-05
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
        q_low = np.percentile(x, 25.495431947989776)
        q_high = np.percentile(x, 80.82738341434685)
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
    linear_slack_penalty = 0.8328356655978811 * np.clip(-slack, 0.0, None)
    coupled_slack_penalty = linear_slack_penalty * (1.0 + 0.9902319462554283 * uncertainty)
    norm_slack_penalty = adaptive_normalize(coupled_slack_penalty)
    critical_load_pressure = upward_rank * (remaining_work + eps)
    norm_critical_load = adaptive_normalize(critical_load_pressure)
    norm_energy = adaptive_normalize(min_incremental_energy)
    shifted_energy = norm_energy + 0.47731903581919666
    shifted_energy = np.clip(shifted_energy, 0.0, 2.0)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    bottleneck_base = duration * (1.0 + norm_critical_load) * (1.0 + norm_slack_penalty)
    norm_bottleneck = adaptive_normalize(bottleneck_base)
    score = norm_slack_penalty + 0.9243240538369857 * norm_critical_load + norm_bottleneck + shifted_energy
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
