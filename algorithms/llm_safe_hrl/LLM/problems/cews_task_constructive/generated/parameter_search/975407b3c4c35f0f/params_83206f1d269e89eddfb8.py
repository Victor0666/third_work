import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Novel priority rule emphasizing deadline risk mitigation, critical-path awareness,
    and energy-duration efficiency — with starvation prevention and uncertainty-coupled urgency.
    
    Key innovations:
      - Piecewise slack handling: strong penalty for negative slack, gentle reward for positive slack
      - Duration-normalized energy efficiency: min_incremental_energy / (min_exec_time + min_comm_time + epsilon)
      - Critical-path bonus gated by upward_rank percentile (not raw value) to avoid dominance
      - Ready-wait boost decays exponentially to prevent over-prioritization of old tasks
      - Uncertainty-slack coupling: amplifies urgency only when both slack is tight AND uncertainty is high
      - All features robustly normalized using configurable IQR percentiles
    """
    eps = 0.002436023327901544
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
        q25, q75 = np.percentile(x, [18.594642241022285, 75.00040339454004])
        iqr = q75 - q25
        center = np.median(x)
        denom = iqr if iqr > eps else np.max(np.abs(x - center)) + eps
        return (x - center) / (denom + eps)
    slack_norm = iqr_normalize(slack)
    slack_penalty = np.where(slack < 0, 10.862403594966889 * -slack_norm, -1.5641514307838669 * slack_norm)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / duration
    energy_eff_norm = iqr_normalize(energy_per_duration)
    rank_norm = iqr_normalize(upward_rank)
    if N == 1:
        rank_percentile = np.array([1.0])
    else:
        sorted_ranks = np.sort(upward_rank)
        rank_idx = np.searchsorted(sorted_ranks, upward_rank, side='right')
        rank_percentile = rank_idx / (N + eps)
    critical_gate = np.where(rank_percentile >= 0.6659955774066147, 1.0, 0.0)
    critical_bonus = 2.593852837672888e-05 * rank_norm * critical_gate
    wait_boost = 1.0 - np.exp(-0.0009569170331240441 * ready_wait_time)
    wait_norm = iqr_normalize(wait_boost)
    unc_norm = iqr_normalize(uncertainty)
    slack_pressure = np.maximum(-slack_norm, 0.0)
    coupled_urgency = 0.4266642710165921 * unc_norm * slack_pressure
    dur_eff_ratio = 1.4439545737102304 * iqr_normalize(duration)
    score = slack_penalty + 0.7955335975828025 * energy_eff_norm - critical_bonus - wait_norm + coupled_urgency + dur_eff_ratio
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score
