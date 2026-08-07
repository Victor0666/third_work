import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule incorporating evidence-backed structural changes:
    - Replaces sigmoid urgency with piecewise linear+convex penalty for stronger DDL violation reduction
    - Gates critical-path bonus by percentile rank (not raw upward_rank) to prevent outlier dominance
    - Uses duration-normalized energy efficiency (energy / (exec + comm + ε)) to align cost with urgency
    - Adds conditional DDL-protection gate activated only when `slack < median(slack)` AND `upward_rank > percentile_threshold`
    - Introduces successor-release delay interaction: `min(remaining_work / (slack + ε), cap)` to unblock bottlenecks under tight deadlines
    - Removes wait-time saturation (evidence shows it's inactive) and redundant sigmoid parameters
    - All normalizations use robust IQR with tunable percentiles; no unbounded growth or unstable functions
    """
    eps = 0.0007446833924876657
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
        q_low = np.percentile(x, 27.571605772230097)
        q_high = np.percentile(x, 74.04587137317274)
        iqr = q_high - q_low
        center = np.median(x)
        denom = iqr if iqr > eps else np.max(np.abs(x - center)) + eps
        return (x - center) / (denom + eps)
    median_slack = np.median(slack)
    slack_deviation = slack - median_slack
    slack_penalty = np.where(slack_deviation >= 0, slack_deviation, slack_deviation ** 2)
    norm_slack_penalty = iqr_normalize(slack_penalty)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / duration
    norm_energy_eff = iqr_normalize(energy_per_duration)
    release_pressure = np.minimum(remaining_work / (slack + eps), 8.218351928772687)
    norm_release_pressure = iqr_normalize(release_pressure)
    if N == 1:
        rank_percentile = np.array([1.0])
    else:
        sorted_ranks = np.sort(upward_rank)
        rank_idx = np.searchsorted(sorted_ranks, upward_rank, side='right')
        rank_percentile = rank_idx / (N + eps)
    critical_gate = np.where(rank_percentile >= 0.7463868853653615, 1.0, 0.0)
    norm_rank = iqr_normalize(upward_rank)
    critical_bonus = critical_gate * norm_rank
    ddl_protection_active = ((slack < median_slack) & (rank_percentile >= 0.6621451175164141)).astype(float)
    norm_uncertainty = iqr_normalize(uncertainty)
    score = norm_slack_penalty + 0.27863055470400155 * norm_release_pressure + 0.5100887027165608 * norm_energy_eff - critical_bonus + ddl_protection_active * norm_uncertainty
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
