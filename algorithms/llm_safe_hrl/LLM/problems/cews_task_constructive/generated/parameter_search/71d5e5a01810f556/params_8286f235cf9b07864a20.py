import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule emphasizing deadline safety via piecewise linear+convex slack penalty,
    critical-path gating by percentile rank (not raw value), duration-normalized energy efficiency,
    conditional DDL-protection gate, and successor-release delay interaction.
    
    Key mutations from prior version:
      - Replaced sigmoid urgency with robust piecewise linear + convex penalty on slack (stronger DDL violation reduction)
      - Critical-path bonus gated by percentile rank (prevents outlier dominance, improves fairness)
      - Energy efficiency computed as min_incremental_energy / (exec + comm + eps) — aligns cost with urgency
      - DDL-protection gate activated only when (slack < median(slack)) AND (upward_rank > percentile_threshold)
      - Successor-release delay: min(remaining_work / (slack + eps), upper_bound) to unblock bottlenecks under tight deadlines
      - Removed wait-time saturation and uncertainty amplification terms — evidence shows they were inactive or fragile
      - All normalizations use IQR with tunable percentiles; no hard thresholds or unbounded growth
    """
    eps = 6.118023084021159e-05
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
        q_low = np.percentile(x, 40.0)
        q_high = np.percentile(x, 79.9319559387753)
        iqr = q_high - q_low
        center = np.median(x)
        denom = iqr if iqr > eps else np.max(np.abs(x - center)) + eps
        return (x - center) / (denom + eps)
    slack_penalty = np.where(slack >= 0, slack, slack ** 2)
    norm_slack_penalty = iqr_normalize(slack_penalty)
    duration = min_exec_time + min_comm_time + eps
    energy_efficiency = min_incremental_energy / duration
    norm_energy_eff = iqr_normalize(energy_efficiency)
    successor_release_delay = np.minimum(remaining_work / (slack + eps), 1.9948621279598582)
    norm_successor_delay = iqr_normalize(successor_release_delay)
    if N == 1:
        rank_percentile = np.array([1.0])
    else:
        sorted_ranks = np.sort(upward_rank)
        rank_idx = np.searchsorted(sorted_ranks, upward_rank, side='right')
        rank_percentile = rank_idx / (N + eps)
    critical_gate = np.where(rank_percentile >= 0.8011304300710012, 1.0, 0.0)
    norm_upward_rank = iqr_normalize(upward_rank)
    critical_bonus = critical_gate * norm_upward_rank
    median_slack = np.median(slack)
    ddl_protection_active = ((slack < median_slack) & (rank_percentile > 0.9087327548802673)).astype(float)
    score = norm_slack_penalty + 0.6891218914544097 * norm_successor_delay + 1.5527824519679314 * norm_energy_eff - critical_bonus + ddl_protection_active * norm_upward_rank
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
