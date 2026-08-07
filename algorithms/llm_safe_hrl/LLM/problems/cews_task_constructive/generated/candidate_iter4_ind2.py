import numpy as np
RULE_METADATA = {'structure_hash': 'c7480e9d30697682e3810375264852883b459ea076871b0dcd1dff6ab8abb577', 'parameter_schema_hash': '4d4dfd3537015e2ac15d3205143c97486aeb87779ec9cdd2cad48602d7bb2898', 'best_parameter_hash': '38b108c96908fa93e6e8d6ee593f3402a8ff8867dce6d6f7789ac77072b3d9b2', 'best_parameters': {'epsilon': 2.008548174002997e-06, 'ddl_protection_gate_threshold': 0.7328540182700787, 'energy_duration_ratio_weight': 1.6597663386886456, 'critical_rank_percentile': 0.7406057179479344, 'iqr_low_percentile': 31.633154377120555, 'iqr_high_percentile': 66.40029985535313, 'successor_bottleneck_coupling': 0.17567178698294414, 'successor_delay_cap': 5.616635224541979, 'slack_convex_coeff': 0.44543867644539703, 'slack_linear_coeff': 2.7526330209767105, 'near_deadline_slope': 0.41529499959823946, 'loose_slack_slope': 0.09820919687648375}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': '0f8ab96cb594764df1d5638dada3f60aa2e55e5db66fae09333de259c338c07f', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule emphasizing deadline risk mitigation via piecewise linear+convex penalty,
    gated critical-path bonus, duration-normalized energy efficiency, and successor-release delay coupling.
    
    Key mutations from prior version:
      - Replaces sigmoid urgency with robust piecewise-linear+convex slack penalty (stronger DDL violation reduction)
      - Gates critical-path bonus by percentile rank (not raw upward_rank) to prevent outlier dominance
      - Uses duration-normalized energy efficiency (energy / (exec + comm + ε)) aligned with urgency scaling
      - Adds conditional DDL-protection gate activated only when (slack < median(slack)) AND (upward_rank > percentile_threshold)
      - Introduces successor-release delay interaction: min(remaining_work / (slack + ε), successor_delay_cap) to unblock bottlenecks under tight deadlines
      - Removes wait-time saturation (evidence shows it's inactive; avoids unnecessary complexity)
      - Removes uncertainty amplification (evidence shows it's inactive; simplifies structure)
      - All normalizations use IQR with tunable percentiles; no hard thresholds or unbounded growth
    """
    eps = 2.008548174002997e-06
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
        q_low = np.percentile(x, 31.633154377120555)
        q_high = np.percentile(x, 66.40029985535313)
        iqr = q_high - q_low
        center = np.median(x)
        denom = iqr if iqr > eps else np.max(np.abs(x - center)) + eps
        return (x - center) / (denom + eps)
    slack_penalty = np.where(slack < 0.0, -slack * 2.7526330209767105 + slack ** 2 * 0.44543867644539703, np.where(slack <= 1.0, slack * 0.41529499959823946, 0.41529499959823946 + (slack - 1.0) * 0.09820919687648375))
    norm_slack_penalty = iqr_normalize(slack_penalty)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / duration
    norm_energy_eff = iqr_normalize(energy_per_duration)
    successor_release_delay = np.minimum(remaining_work / (slack + eps), 5.616635224541979)
    norm_successor_delay = iqr_normalize(successor_release_delay)
    if N == 1:
        rank_percentile = np.array([1.0])
    else:
        sorted_ranks = np.sort(upward_rank)
        rank_idx = np.searchsorted(sorted_ranks, upward_rank, side='right')
        rank_percentile = rank_idx / (N + eps)
    critical_gate = np.where(rank_percentile >= 0.7406057179479344, 1.0, 0.0)
    norm_rank = iqr_normalize(upward_rank)
    critical_bonus = critical_gate * norm_rank
    median_slack = np.median(slack)
    max_uncertainty = np.max(uncertainty) if N > 0 else 1.0
    ddl_protection_active = ((slack < median_slack) & (uncertainty > 0.7328540182700787 * max_uncertainty)).astype(float)
    score = norm_slack_penalty + 0.17567178698294414 * norm_successor_delay + 1.6597663386886456 * norm_energy_eff - critical_bonus + ddl_protection_active * norm_rank
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
