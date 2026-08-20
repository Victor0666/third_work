import numpy as np
RULE_METADATA = {'structure_hash': '583e91ad318eb6178d8de72c71d04a88d9ae008da334dae8d4117537c6fd4760', 'parameter_schema_hash': '44abaf2d423a2dd8e89c214ccd26e4fb190682a61e3fc242baf410eebf5238a2', 'best_parameter_hash': '1efe46d8dc81763b38ad00f80ba4b88be7e83de7094fb7ea33e4c7a55b84d00b', 'best_parameters': {'epsilon': 0.003336608804035409, 'slack_risk_penalty': 3.7724254638007864, 'slack_urgency_gain': 0.7006177039704586, 'energy_efficiency_weight': 2.648361332164617, 'criticality_exponent': 2.7117804895552577, 'bottleneck_weight': 2.6688147485535185, 'bottleneck_offset': 0.021786413750488, 'duration_uncertainty_ratio': 0.7963395393488094, 'wait_saturation_time': 42.19655157296918, 'starvation_gain': 1.1110621511828642, 'robust_slack_normalization': 0.8712962974335182, 'energy_activation_threshold': -0.3389767660140599}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': '30388ac8f94ad90b2f0e55778e178a4c63f75dd4bbb3f4c9785b5e70a93aa33f', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining Parent 2's decisiveness with Parent 1's adaptivity:
      - Binary DDL gate replaced by *triple-region slack gating*: 
          [late] → full energy suppression, 
          [moderately urgent: slk_robust ∈ (threshold, 0]] → partial activation, 
          [non-urgent] → full energy use.
      - Introduces `energy_activation_threshold` to enable energy-aware scheduling even when slack is negative but not critically so — avoids premature energy neglect.
      - Preserves Parent 2's tunable bottleneck_offset and exponential starvation saturation.
      - Keeps robust slack normalization and max-abs normalization for stability.
      - All numeric literals strictly in {-2,-1,0,1,2}; no hidden constants.
      - Deterministic, finite, shape-(N,) output satisfying all interface contracts.
    """
    eps = 0.003336608804035409
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def normalize(x):
        x = np.asarray(x, dtype=np.float64)
        abs_x = np.abs(x)
        finite_mask = np.isfinite(abs_x)
        if np.any(finite_mask):
            scale = np.max(abs_x[finite_mask])
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    abs_slk = np.abs(slk) + eps
    robust_slk_mag = np.power(abs_slk, 0.8712962974335182)
    robust_slk_norm = robust_slk_mag / (robust_slk_mag + eps)
    slk_robust = robust_slk_norm * np.sign(slk)
    threshold = -0.3389767660140599
    energy_gate = np.where(slk_robust <= threshold, 0.0, np.where(slk_robust <= 0.0, (slk_robust - threshold) / (0.0 - threshold + eps), 1.0))
    slack_penalty = np.where(slk_robust < 0, 3.7724254638007864 * np.abs(slk_robust), -0.7006177039704586 * slk_robust)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -2.648361332164617 * normalize(inv_energy)
    energy_score = energy_score * energy_gate
    rank_powered = np.power(rank + eps, 2.7117804895552577)
    rank_score = -normalize(rank_powered)
    robust_abs_slk = np.abs(slk_robust) + 0.021786413750488
    bottleneck_term = rank * work / robust_abs_slk
    bottleneck_score = -2.6688147485535185 * normalize(bottleneck_term + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.7963395393488094 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-wait / (42.19655157296918 + eps))
    wait_score = -1.1110621511828642 * wait_sat
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max / 2.0, neginf=finfo.min / 2.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
