import numpy as np
RULE_METADATA = {'structure_hash': '1fa7a1aa31ff2570a97a19bbf1b339857ca03ad7c8c042dd037131f510ee9875', 'parameter_schema_hash': 'e3e8c988f3c5aa2e97c78b5404d5465bf766c405dab956fc4ee79adb6f7bd610', 'best_parameter_hash': '12cbcf1cb8cc6848e06a45c75f9cea7b9e5e1b56e0bfdad1cbf7a861efa3b8b3', 'best_parameters': {'epsilon': 0.07627531367301765, 'slack_linear_penalty_slope': 5.21582906764132, 'slack_linear_reward_slope': 2.930114184222592, 'critical_rank_percentile_threshold': 0.663302019626526, 'successor_delay_weight': 2.223988707014001, 'energy_duration_load_adjustment': 0.8553524238969356, 'load_proxy_factor': 0.04806081480644747, 'wait_fairness_exponent': 0.7202577239941488, 'iqr_percentile_low': 18.5708611606562, 'iqr_percentile_high': 86.58382114673476, 'successor_delay_upper_bound': 18.139029960940757}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': '71c59a9883f59ceb4f2e67f64f0f9e6c326a56c7cbe19248bc3c4732e24a450c', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule restoring sharp deadline-risk discrimination via piecewise-linear slack,
       reintroducing percentile-gated critical-path bonus, and adding load-aware energy efficiency.
    
    Key improvements:
      - Piecewise-linear slack handling: exact linear penalty for negative slack, linear reward for positive slack
        → preserves hard DDL violation sensitivity lost in sigmoid approximation.
      - Critical-path bonus activated only above percentile threshold (not raw rank) → robust to outliers.
      - Load-aware energy efficiency: adjusts marginal energy by `1 + load_proxy_factor * uncertainty`,
        modeling congestion-induced inefficiency without requiring explicit load telemetry.
      - Power-law wait boost (`ready_wait_time ** exponent`) → tunable fairness-staleness tradeoff.
      - All features normalized via adaptive IQR percentiles for outlier resilience.
    """
    eps = 0.07627531367301765
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
        q_low = np.percentile(x, 18.5708611606562)
        q_high = np.percentile(x, 86.58382114673476)
        iqr = q_high - q_low
        center = np.median(x)
        denom = iqr if iqr > eps else np.max(np.abs(x - center)) + eps
        return (x - center) / (denom + eps)
    slack_norm = iqr_normalize(slack)
    slack_score = np.where(slack < 0, 5.21582906764132 * -slack_norm, -2.930114184222592 * slack_norm)
    successor_delay = np.clip(remaining_work / (slack + eps), 0.0, 18.139029960940757)
    norm_successor_delay = iqr_normalize(successor_delay)
    if N == 1:
        rank_percentile = np.array([1.0])
    else:
        sorted_ranks = np.sort(upward_rank)
        rank_idx = np.searchsorted(sorted_ranks, upward_rank, side='right')
        rank_percentile = rank_idx / (N + eps)
    critical_gate = np.where(rank_percentile >= 0.663302019626526, 1.0, 0.0)
    norm_upward_rank = iqr_normalize(upward_rank)
    critical_bonus = norm_upward_rank * critical_gate
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    load_adjustment = 1.0 + 0.04806081480644747 * uncertainty
    energy_per_duration = min_incremental_energy / (duration * load_adjustment + eps)
    norm_energy_eff = iqr_normalize(energy_per_duration)
    wait_boost = np.power(ready_wait_time + eps, 0.7202577239941488)
    norm_wait = iqr_normalize(wait_boost)
    score = slack_score + 2.223988707014001 * norm_successor_delay - critical_bonus + 0.8553524238969356 * norm_energy_eff - norm_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
