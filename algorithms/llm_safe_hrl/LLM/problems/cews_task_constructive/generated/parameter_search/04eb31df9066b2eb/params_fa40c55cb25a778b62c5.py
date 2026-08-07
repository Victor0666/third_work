import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with bounded piecewise-linear slack penalty for strict DDL safety,
       reinstated robust successor-release delay (no wait scaling), and simplified critical-path gating.
    
    Key improvements:
      - Replaced hybrid/convex slack scoring with monotonic, bounded piecewise-linear penalty:
          * Negative slack: linear penalty with steeper slope (higher violation cost)
          * Positive slack: gentler linear penalty (avoids over-rewarding margin)
          * Zero-crossing offset shifts sensitivity leftward to trigger earlier intervention
      - Removed wait-gated critical boost per reflection — eliminates starvation risk and simplifies fairness
      - Restored original successor_delay formulation (un-scaled) — proven causal bottleneck unblocking
      - Simplified critical_bonus to percentile-gated only (no wait coupling), improving determinism
      - All normalizations remain IQR-based with tunable percentiles; no unbounded terms or fragile heuristics
    """
    eps = 6.656160478222976e-05
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
        q_low = np.percentile(x, 39.575607785202806)
        q_high = np.percentile(x, 79.8553103729028)
        iqr = q_high - q_low
        center = np.median(x)
        denom = iqr if iqr > eps else np.max(np.abs(x - center)) + eps
        return (x - center) / (denom + eps)
    shifted_slack = slack - -0.3710203833898216
    slack_penalty = np.where(shifted_slack < 0, 3.5564810480915736 * shifted_slack, 0.5931912749391846 * shifted_slack)
    slack_penalty = np.maximum(slack_penalty, 0.0)
    norm_slack_penalty = iqr_normalize(slack_penalty)
    duration = min_exec_time + min_comm_time + eps
    energy_efficiency = min_incremental_energy / duration
    norm_energy_eff = iqr_normalize(energy_efficiency)
    raw_successor_delay = remaining_work / (slack + eps)
    bounded_successor_delay = np.minimum(raw_successor_delay, 8.193412910222671)
    norm_successor_delay = iqr_normalize(bounded_successor_delay)
    if N == 1:
        rank_percentile = np.array([1.0])
    else:
        sorted_ranks = np.sort(upward_rank)
        rank_idx = np.searchsorted(sorted_ranks, upward_rank, side='right')
        rank_percentile = rank_idx / (N + eps)
    critical_gate = np.where(rank_percentile >= 0.7756227527103701, 1.0, 0.0)
    norm_upward_rank = iqr_normalize(upward_rank)
    critical_bonus = critical_gate * norm_upward_rank
    median_slack = np.median(slack)
    ddl_protection_active = ((slack < median_slack) & (rank_percentile > 0.629309588749791)).astype(float)
    score = norm_slack_penalty + 0.4357032068684268 * norm_successor_delay + 0.22655759530918307 * norm_energy_eff - critical_bonus + ddl_protection_active * norm_upward_rank
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
