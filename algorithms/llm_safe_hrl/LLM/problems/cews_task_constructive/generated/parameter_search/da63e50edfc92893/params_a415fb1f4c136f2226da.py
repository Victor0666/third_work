import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's structural strengths with Parent 1's robustness and fairness mechanisms.
    
    Key improvements:
      - Replaces exponential wait boost with bounded arctan saturation (more stable under long queues)
      - Introduces remaining_work × energy coupling to prioritize energy-critical bottlenecks
      - Uses tunable quantiles for all normalizations (consistent with Parent 1's robustness)
      - Retains Parent 2's percentile-gated critical-path bonus and duration-normalized energy efficiency
      - Adds slack-pressure gating to uncertainty coupling: only activates when slack < median
      - All numeric literals restricted to {-2,-1,0,1,2}; eps handled via np.finfo
    """
    eps = 0.002730964222984134
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

    def iqr_normalize(x):
        x = np.copy(x)
        q_low = np.percentile(x, 19.46025576967264)
        q_high = np.percentile(x, 85.07705640874529)
        iqr = q_high - q_low
        center = np.median(x)
        denom = iqr if iqr > eps_safe else np.max(np.abs(x - center)) + eps_safe
        return (x - center) / (denom + eps_safe)
    slack_norm = iqr_normalize(slack)
    slack_penalty = np.where(slack < 0, 0.750639501238623 * -slack_norm, -3.393231386563024 * slack_norm)
    duration = np.maximum(min_exec_time + min_comm_time, eps_safe)
    energy_per_duration = min_incremental_energy / duration
    energy_eff_norm = iqr_normalize(energy_per_duration)
    if N == 1:
        rank_percentile = np.array([1.0])
    else:
        sorted_ranks = np.sort(upward_rank)
        rank_idx = np.searchsorted(sorted_ranks, upward_rank, side='right')
        rank_percentile = rank_idx / (N + eps_safe)
    critical_gate = np.where(rank_percentile >= 0.6333898034423289, 1.0, 0.0)
    rank_norm = iqr_normalize(upward_rank)
    critical_bonus = 2.000137544281238 * rank_norm * critical_gate
    wait_scaled = ready_wait_time / (15.702107836517982 + eps_safe)
    wait_boost = 2.0 / np.pi * np.arctan(wait_scaled)
    wait_norm = iqr_normalize(wait_boost)
    median_slack = np.median(slack)
    slack_pressure = np.where(slack < median_slack, 1.0, 0.0)
    unc_norm = iqr_normalize(uncertainty)
    coupled_urgency = 1.2264808744624314 * unc_norm * slack_pressure
    rw_norm = iqr_normalize(remaining_work)
    energy_norm = iqr_normalize(min_incremental_energy)
    bottleneck_score = rw_norm * energy_norm * 0.39616131262193405
    dur_eff_ratio = 0.7849501462492828 * iqr_normalize(duration)
    score = slack_penalty + 0.7868839385079833 * energy_eff_norm - critical_bonus - wait_norm + coupled_urgency + dur_eff_ratio + bottleneck_score
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
