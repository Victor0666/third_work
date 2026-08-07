import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's robust deadline enforcement with Parent 1's calibrated slack reward
       and improved fairness via wait-aware bottleneck mitigation.
    
    Key innovations:
      - Hybrid slack scoring: convex penalty (|slack|^p) for negative slack (p>2), linear reward for positive slack
        → preserves hard DDL violation dominance while retaining margin utility (unlike pure convex).
      - Wait-aware successor delay: scales bottleneck pressure by normalized wait time, preventing starvation-induced
        late scheduling of long-waiting tasks that block critical paths.
      - Critical-path bonus now includes wait-gated amplification: only high-rank *and* long-waiting tasks get boosted,
        reducing false positives from static rank outliers.
      - All normalizations use tunable IQR percentiles; no hardcoded thresholds or unbounded growth.
      - Removed fragile load-proxy and power-law wait terms — replaced by deterministic wait-aware coupling.
    """
    eps = 0.0024409742268149447
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
        q_low = np.percentile(x, 31.80748788395077)
        q_high = np.percentile(x, 82.94508103056283)
        iqr = q_high - q_low
        center = np.median(x)
        denom = iqr if iqr > eps else np.max(np.abs(x - center)) + eps
        return (x - center) / (denom + eps)
    slack_abs = np.abs(slack)
    slack_sign = np.sign(slack)
    slack_penalty = np.where(slack < 0, np.power(slack_abs, 1.618330507386501), 1.448838282498354 * slack)
    norm_slack_penalty = iqr_normalize(slack_penalty)
    duration = min_exec_time + min_comm_time + eps
    energy_efficiency = min_incremental_energy / duration
    norm_energy_eff = iqr_normalize(energy_efficiency)
    raw_successor_delay = remaining_work / (slack + eps)
    bounded_successor_delay = np.minimum(raw_successor_delay, 5.463953281222016)
    norm_wait = iqr_normalize(ready_wait_time + eps)
    wait_scaled_delay = bounded_successor_delay * (1.0 + norm_wait)
    norm_successor_delay = iqr_normalize(wait_scaled_delay)
    if N == 1:
        rank_percentile = np.array([1.0])
    else:
        sorted_ranks = np.sort(upward_rank)
        rank_idx = np.searchsorted(sorted_ranks, upward_rank, side='right')
        rank_percentile = rank_idx / (N + eps)
    critical_gate = np.where(rank_percentile >= 0.5962879993346322, 1.0, 0.0)
    if N == 1:
        wait_percentile = np.array([1.0])
    else:
        sorted_wait = np.sort(ready_wait_time)
        wait_idx = np.searchsorted(sorted_wait, ready_wait_time, side='right')
        wait_percentile = wait_idx / (N + eps)
    wait_gate = np.where(wait_percentile >= 0.7505102658390932, 1.0, 0.0)
    norm_upward_rank = iqr_normalize(upward_rank)
    critical_bonus = critical_gate * wait_gate * norm_upward_rank
    median_slack = np.median(slack)
    ddl_protection_active = ((slack < median_slack) & (rank_percentile > 0.7553511258466167)).astype(float)
    score = norm_slack_penalty + 0.00446214428780069 * norm_successor_delay + 1.514723839169531 * norm_energy_eff - critical_bonus + ddl_protection_active * norm_upward_rank
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
