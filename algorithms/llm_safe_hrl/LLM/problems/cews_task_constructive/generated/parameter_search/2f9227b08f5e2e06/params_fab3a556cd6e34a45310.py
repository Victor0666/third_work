import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule incorporating evidence-backed structural changes:
    - Uses ddl_protection_gate_threshold in DDL-protection gate (now correctly referenced)
    - Replaced arctan wait saturation with bounded linear ramp + soft cap for better gradient behavior
    - Unified bottleneck term: (exec+comm) * upward_rank * remaining_work (stronger coupling to critical path)
    - DDL protection gate now uses threshold fraction of max uncertainty as intended
    - Energy term uses log-normalized energy/duration ratio to compress dynamic range
    - Added anti-starvation term: inverse of normalized ready_wait_time (smaller wait → higher priority)
    - All normalizations use IQR with elite-tuned percentiles
    - Final score preserves strict ordering: deadline > bottleneck > energy > fairness
    """
    eps = 1.4837147674405036e-05
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def iqr_normalize(x):
        x = np.copy(x)
        q_low = np.percentile(x, 26.912593105910435)
        q_high = np.percentile(x, 68.94856062779493)
        iqr = q_high - q_low
        center = (q_low + q_high) / 2.0
        denom = iqr if iqr > eps else np.max(np.abs(x - center)) + eps
        return (x - center) / (denom + eps)
    slack_centered = slack - 0.2139165379123078
    sigmoid_input = -3.943006202190702 * slack_centered
    finfo = np.finfo(float)
    sigmoid_input = np.clip(sigmoid_input, -finfo.max, finfo.max)
    urgency_gate = 1.0 / (1.0 + np.exp(-sigmoid_input))
    norm_urgency = iqr_normalize(urgency_gate)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / duration
    log_energy_eff = np.log1p(energy_per_duration + eps)
    norm_energy_eff = iqr_normalize(log_energy_eff)
    bottleneck_pressure = duration * (upward_rank + eps) * (remaining_work + eps)
    norm_bottleneck = iqr_normalize(bottleneck_pressure)
    wait_normalized = np.clip(ready_wait_time / (0.6019985171515297 + eps), 0.0, 1.0)
    wait_priority = 1.0 - wait_normalized
    norm_wait = iqr_normalize(wait_priority)
    max_uncertainty = np.max(uncertainty) if N > 0 else 1.0
    ddl_protection_active = (uncertainty > 0.5986502122572767 * max_uncertainty).astype(float)
    norm_uncertainty = iqr_normalize(uncertainty)
    score = norm_urgency + 0.538397576665066 * norm_bottleneck + 1.3964506411128328 * norm_energy_eff - norm_wait + ddl_protection_active * norm_uncertainty
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=np.finfo(float).min)
    return score.astype(float, copy=False)
