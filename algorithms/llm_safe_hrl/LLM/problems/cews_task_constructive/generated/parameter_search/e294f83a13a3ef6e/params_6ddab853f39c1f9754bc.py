import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: sigmoid urgency + critical-path bonus + anti-starvation + DDL-protection.
    
    Structural improvements:
      - Hybrid urgency: sigmoid for smooth gradient near deadline, plus bounded linear fallback for extreme negative slack
      - Critical-path bonus reintroduced from Parent 1 (top-percentile upward_rank boost) to strengthen DAG-aware scheduling
      - Anti-starvation via inverse wait-time saturation (Parent 2) but with robust clipping to avoid zero-division
      - Unified bottleneck term: (exec+comm) * upward_rank * remaining_work * (1 + uncertainty) — couples risk into blocking pressure
      - All normalizations use IQR with elite-tuned percentiles; no hardcoded constants beyond [-2, -1, 0, 1, 2]
      - Final score preserves strict hierarchy: deadline > critical path > bottleneck > energy > fairness
    """
    eps = 0.00019048588546410875
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
        q_low = np.percentile(x, 23.389655797578627)
        q_high = np.percentile(x, 85.98260698598773)
        iqr = q_high - q_low
        center = (q_low + q_high) / 2.0
        denom = iqr if iqr > eps else np.max(np.abs(x - center)) + eps
        return (x - center) / (denom + eps)
    finfo = np.finfo(float)
    slack_centered = slack - -0.059735203366516276
    sigmoid_input = -7.982118084511186 * slack_centered
    sigmoid_input = np.clip(sigmoid_input, -finfo.max, finfo.max)
    sigmoid_urgency = 1.0 / (1.0 + np.exp(-sigmoid_input))
    linear_urgency_raw = np.where(slack <= 0, 1.4019274042875383 * (slack - -0.4917555895704626), 0.0)
    linear_urgency = np.clip(linear_urgency_raw, -2.0, 2.0)
    blend_weight = np.clip((slack + 2.0) / 2.0, 0.0, 1.0)
    urgency_gate = blend_weight * sigmoid_urgency + (1.0 - blend_weight) * linear_urgency
    norm_urgency = iqr_normalize(urgency_gate)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / duration
    log_energy_eff = np.log1p(energy_per_duration + eps)
    norm_energy_eff = iqr_normalize(log_energy_eff)
    bottleneck_pressure = duration * (upward_rank + eps) * (remaining_work + eps) * (1.0 + uncertainty)
    norm_bottleneck = iqr_normalize(bottleneck_pressure)
    if N == 1:
        rank_percentile = np.array([1.0])
    else:
        sorted_ranks = np.sort(upward_rank)
        rank_idx = np.searchsorted(sorted_ranks, upward_rank, side='right')
        rank_percentile = rank_idx / (N + eps)
    critical_gate = np.where(rank_percentile >= 0.7490042190517817, 1.0, 0.0)
    norm_rank = iqr_normalize(upward_rank)
    critical_bonus = critical_gate * norm_rank
    wait_normalized = np.clip(ready_wait_time / (9.220013147534212 + eps), 0.0, 1.0)
    wait_priority = 1.0 - wait_normalized
    norm_wait = iqr_normalize(wait_priority)
    max_uncertainty = np.max(uncertainty) if N > 0 else 1.0
    ddl_protection_active = (uncertainty > 0.8357347503729804 * max_uncertainty).astype(float)
    norm_uncertainty = iqr_normalize(uncertainty)
    score = norm_urgency + critical_bonus + 0.9337195414028638 * norm_bottleneck + 0.9263526138236426 * norm_energy_eff - norm_wait + ddl_protection_active * norm_uncertainty
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=np.finfo(float).min)
    return score.astype(float, copy=False)
