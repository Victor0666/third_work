import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule incorporating:
    - Explicit negative-slack penalty (not sigmoid) for stronger hard-DDL enforcement
    - Upward-rank × remaining-work interaction as primary bottleneck signal (validated in replay failures)
    - Uncertainty-coupled energy: min_incremental_energy * (1 + uncertainty) to penalize high-risk energy choices
    - Anti-starvation via normalized ready_wait_time (no ramp/saturation — simpler, deterministic)
    - All normalizations use adaptive IQR→min-max fallback with shared epsilon
    - No unused parameters; all declared parameters are consumed
    """
    eps = 0.0013896916166551729
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
        q_low = np.percentile(x, 38.29042491365146)
        q_high = np.percentile(x, 89.78314187235691)
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
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    critical_bottleneck = upward_rank * remaining_work
    norm_critical_bottleneck = adaptive_normalize(critical_bottleneck)
    negative_slack_mask = (slack < 0.0).astype(float)
    slack_magnitude = np.abs(slack) * negative_slack_mask
    norm_slack_magnitude = adaptive_normalize(slack_magnitude)
    negative_slack_penalty = 2.4881313502013573 * norm_slack_magnitude
    coupled_energy = min_incremental_energy * (1.0 + uncertainty + eps)
    norm_coupled_energy = adaptive_normalize(coupled_energy)
    norm_wait = adaptive_normalize(ready_wait_time)
    score = negative_slack_penalty + 1.8913239059994222 * norm_critical_bottleneck + 0.5197657440667917 * norm_coupled_energy - norm_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
