import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's superior deadline/energy structure with Parent 1's robustness and starvation control.
    
    Key structural improvements:
      - Replaces exponential wait boost with bounded arctan saturation (from Parent 1) for smoother fairness control
      - Adds remaining_work × energy coupling to penalize high-energy tasks blocking large downstream work
      - Uses robust percentile-based criticality gate (Parent 2) but adds slack-aware activation: only triggers when slack < median
      - Combines piecewise slack handling (Parent 2) with uncertainty-amplified penalty (Parent 1 style)
      - All normalizations use tunable IQR percentiles; no hardcoded quantiles
    """
    eps = 0.00542235590390615
    N = len(slack)
    finfo = np.finfo(float)
    eps_safe = max(eps, finfo.tiny)
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
        q_low = np.percentile(x, 16.60204714302079)
        q_high = np.percentile(x, 66.19592902404572)
        iqr = q_high - q_low
        center = np.median(x)
        denom = iqr if iqr > eps_safe else np.max(np.abs(x - center)) + eps_safe
        return (x - center) / (denom + eps_safe)
    slack_norm = iqr_normalize(slack)
    slack_penalty = np.where(slack < 0, 4.9365519275664305 * -slack_norm, -3.8481360729926277 * slack_norm)
    duration = np.maximum(min_exec_time + min_comm_time, eps_safe)
    energy_per_duration = min_incremental_energy / duration
    energy_eff_norm = iqr_normalize(energy_per_duration)
    rank_norm = iqr_normalize(upward_rank)
    if N == 1:
        rank_percentile = np.array([1.0])
    else:
        sorted_ranks = np.sort(upward_rank)
        rank_idx = np.searchsorted(sorted_ranks, upward_rank, side='right')
        rank_percentile = rank_idx / (N + eps_safe)
    median_slack = np.median(slack)
    critical_gate = np.where((rank_percentile >= 0.6368642413898399) & (slack < median_slack), 1.0, 0.0)
    critical_bonus = 0.27477615922380616 * rank_norm * critical_gate
    wait_scaled = ready_wait_time / (0.07747151745485337 + eps_safe)
    wait_boost = 2.0 / np.pi * np.arctan(wait_scaled)
    wait_norm = iqr_normalize(wait_boost)
    unc_norm = iqr_normalize(uncertainty)
    slack_pressure = np.maximum(-slack_norm, 0.0)
    coupled_urgency = 0.16881927004631792 * unc_norm * slack_pressure
    dur_eff_ratio = 0.03316506023026674 * iqr_normalize(duration)
    rw_norm = iqr_normalize(remaining_work)
    energy_norm = iqr_normalize(min_incremental_energy)
    rw_energy_coupling = np.clip(0.4893284028180003 * rw_norm * energy_norm, -2.0, 2.0)
    score = slack_penalty + 0.5825413877099535 * energy_eff_norm - critical_bonus - wait_norm + coupled_urgency + dur_eff_ratio + rw_energy_coupling
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
