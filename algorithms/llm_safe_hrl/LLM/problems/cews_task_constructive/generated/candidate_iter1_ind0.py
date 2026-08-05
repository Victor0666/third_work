import numpy as np
RULE_METADATA = {'structure_hash': '1968a877e44f5f5357fabe41caca604468b435f3e0a128f95c053e08073752e0', 'parameter_schema_hash': 'f3957b8c9e75e250221897ab284aabfb209d886b64fff5d9dd5541e9e97ab293', 'best_parameter_hash': '21ef533e3efbe0dfa142ae07d5bc4fe59bcb6906e45cae1348bd91ce798c0686', 'best_parameters': {'epsilon': 0.0003295324196489967, 'slack_risk_exponent': 2.358131613529369, 'critical_path_boost': 1.8516489111718133, 'energy_efficiency_ratio_weight': 1.2039795250061809, 'uncertainty_slack_coupling': 1.3641692778353078, 'wait_saturation_threshold': 1.1012076253905576, 'duration_uncertainty_penalty': 0.8205348845176695, 'rank_normalized_gap_weight': 1.844869761325997, 'nan_replacement': 877799.9209503441, 'posinf_replacement': 16379.879403958394, 'neginf_replacement': -2267329.8681353573}, 'optimizer_config_hash': '8e89e2812aaa16a6bfeea9b77a3951401528f16bd6e28802475942beac19e1e1', 'parameter_diagnostics_hash': '1a1344b7ae03b50e39d314eefa335cd9364806b1edfd587021335a2ac4e101d2', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Novel priority rule emphasizing deadline risk nonlinearity, critical-path balance,
    energy efficiency per work, and saturation-based anti-starvation.
    
    Key innovations:
      - Slack penalty uses clipped power law: (max(0, -slack))^exponent → strong but bounded urgency
      - Critical path boost applies only above median rank → avoids over-prioritizing all high-rank tasks
      - Energy efficiency ratio = energy / (work + eps) → favors low-energy-per-MI tasks
      - Duration uncertainty penalty = std(duration)/mean(duration+eps) → penalizes volatile execution/comm
      - Ready wait uses smooth saturation: 1 - exp(-wait/threshold) → anti-starvation with diminishing returns
      - Rank gap term promotes diversity: prioritizes tasks *below* max rank to avoid congestion on single path
    """
    eps = 0.0003295324196489967
    N = len(min_exec_time)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def robust_normalize(x):
        x_abs = np.abs(x)
        x_mean = np.mean(x_abs)
        scale = x_mean + eps
        return x / scale
    neg_slack = np.maximum(-slack, 0.0)
    slack_risk = np.power(neg_slack + eps, 2.358131613529369)
    slack_risk_norm = robust_normalize(slack_risk)
    rank_median = np.median(upward_rank)
    rank_boost_mask = (upward_rank >= rank_median).astype(float)
    rank_boosted = upward_rank * (1.0 + 1.8516489111718133 * rank_boost_mask)
    rank_score = -robust_normalize(rank_boosted)
    energy_per_work = min_incremental_energy / (remaining_work + eps)
    energy_efficiency_score = robust_normalize(energy_per_work)
    duration = min_exec_time + min_comm_time
    duration_mean = np.mean(duration) + eps
    duration_std = np.std(duration)
    duration_uncert_ratio = duration_std / duration_mean
    duration_uncert_penalty = 0.8205348845176695 * np.full(N, duration_uncert_ratio)
    wait_saturation = 1.0 - np.exp(-ready_wait_time / (1.1012076253905576 + eps))
    wait_score = -robust_normalize(wait_saturation)
    slack_pressure_indicator = (neg_slack > eps).astype(float)
    coupled_risk = uncertainty * slack_pressure_indicator
    coupled_risk_norm = robust_normalize(coupled_risk)
    rank_max = np.max(upward_rank) + eps
    rank_gap = (rank_max - upward_rank) / rank_max
    rank_gap_score = -robust_normalize(rank_gap) * 1.844869761325997
    score = slack_risk_norm * 1.0 + energy_efficiency_score * 1.2039795250061809 + rank_score * 1.0 + duration_uncert_penalty * 1.0 + wait_score * 1.0 + coupled_risk_norm * 1.3641692778353078 + rank_gap_score * 1.0
    score = np.nan_to_num(score, nan=877799.9209503441, posinf=16379.879403958394, neginf=-2267329.8681353573)
    return score
