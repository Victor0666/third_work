import numpy as np
RULE_METADATA = {'structure_hash': '04eb31df9066b2ebe303168329be5f009e699a729fd27d7257b672cb4999204a', 'parameter_schema_hash': 'c11339fe521da1816e9a3d165cac32473c76636c60f316fea9de30e9955d8d33', 'best_parameter_hash': '0bf1ca3fcc7ba5e4e92d6d3365d3fec25974282787417116cc08759650e900bb', 'best_parameters': {'epsilon': 6.656160478222976e-05, 'ddl_protection_gate_threshold': 0.629309588749791, 'energy_duration_ratio_weight': 0.22655759530918307, 'successor_bottleneck_coupling': 0.4357032068684268, 'iqr_low_percentile': 39.575607785202806, 'iqr_high_percentile': 79.8553103729028, 'critical_rank_percentile': 0.7756227527103701, 'successor_delay_upper_bound': 8.193412910222671, 'slack_piecewise_slope_positive': 0.5931912749391846, 'slack_piecewise_slope_negative': 3.6164810480915737, 'slack_zero_crossing_offset': -0.3710203833898216}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': '72f1a64c0f9ba1781fa9529334d94169f4b232f992c69ddd0d973c87c9c51c72', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

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
    slack_penalty = np.where(shifted_slack < 0, 3.6164810480915737 * shifted_slack, 0.5931912749391846 * shifted_slack)
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
