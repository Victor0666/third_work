import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule with unified quantile normalization, joint criticality gating,
    uncertainty-coupled energy penalty, and slack-gated wait boost — all within 12 parameters.
    
    Key features:
      - Robust quantile-based normalization (tunable low/high) for all features
      - Criticality gate requires *both* top-percentile upward_rank *and* tight slack (< median)
      - Uncertainty-coupled energy penalty: only active when energy_eff and uncertainty are both high
      - Wait boost enabled only for non-negative slack (avoids starving overdue tasks)
      - All numeric literals are {-2,-1,0,1,2}; eps handled via PARAMS and np.finfo
    """
    eps = 0.0001245759643277049
    finfo = np.finfo(float)
    eps_safe = max(eps, finfo.tiny)
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def robust_normalize(x):
        x = np.copy(x)
        q1 = np.quantile(x, 0.1736845576330768)
        q3 = np.quantile(x, 0.7820885612072597)
        iqr = q3 - q1
        center = np.median(x)
        scale = iqr if iqr > eps_safe else eps_safe
        return (x - center) / (scale + eps_safe)
    duration = min_exec_time + min_comm_time
    norm_duration = robust_normalize(duration)
    slack_norm = robust_normalize(slack)
    slack_penalty = np.where(slack < 0, 4.207374553617227 * -slack_norm, -4.452466219866493 * slack_norm)
    duration_safe = np.maximum(duration, eps_safe)
    energy_per_duration = min_incremental_energy / duration_safe
    norm_energy_eff = robust_normalize(energy_per_duration)
    median_unc = np.median(uncertainty)
    median_energy_eff = np.median(energy_per_duration)
    unc_norm = robust_normalize(uncertainty)
    energy_uncertainty_penalty = np.where((energy_per_duration > median_energy_eff) & (uncertainty > median_unc), unc_norm * norm_energy_eff * 1.029941509102875, 0.0)
    if N == 1:
        rank_percentile = np.array([1.0])
    else:
        sorted_ranks = np.sort(upward_rank)
        rank_idx = np.searchsorted(sorted_ranks, upward_rank, side='right')
        rank_percentile = rank_idx / (N + eps_safe)
    median_slack = np.median(slack)
    critical_gate = ((rank_percentile >= 0.6589761259769454) & (slack < median_slack)).astype(float)
    norm_rank = robust_normalize(upward_rank + 0.14567391351169437)
    critical_bonus = 1.2534292801094957 * norm_rank * critical_gate
    wait_boost = np.where(slack >= 0, 1.0 - np.exp(-0.0010129166210997326 * ready_wait_time), 0.0)
    norm_wait = robust_normalize(wait_boost)
    coupled_urgency = np.where((slack < 0) & (uncertainty > median_unc), unc_norm * -slack_norm * 1.029941509102875, 0.0)
    dur_eff_ratio = 1.125824711830579 * norm_duration
    score = slack_penalty + 2.5425313914022993 * norm_energy_eff + energy_uncertainty_penalty - critical_bonus - norm_wait + coupled_urgency + dur_eff_ratio
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
