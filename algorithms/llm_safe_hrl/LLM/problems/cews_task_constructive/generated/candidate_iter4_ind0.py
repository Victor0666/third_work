import numpy as np
RULE_METADATA = {'structure_hash': '8d90a86f1471914de42f52cd15f74d31a8dd216dac6836a087537a217d2b4816', 'parameter_schema_hash': '846ceaba82a431946623553dfb7926ce1946b72f8409436ac155ff0c9e01b1db', 'best_parameter_hash': '18d2ae5cf4822e01542f84653e83076055e5317b64a8b607baf06b411bbb9134', 'best_parameters': {'epsilon': 0.0007446833924876657, 'ddl_protection_gate_threshold': 0.6621451175164141, 'energy_duration_ratio_weight': 0.5100887027165608, 'successor_bottleneck_coupling': 0.27863055470400155, 'iqr_low_percentile': 27.571605772230097, 'iqr_high_percentile': 73.44587137317275, 'critical_rank_percentile': 0.7463868853653615, 'successor_release_cap': 8.218351928772687}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': '4af1c9b410e08ff20beba1220e93560f1faf4849a592362798890199665b96fe', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

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
        q_high = np.percentile(x, 73.44587137317275)
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
