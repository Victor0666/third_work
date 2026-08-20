import numpy as np
RULE_METADATA = {'structure_hash': 'b57cf717d06a2574cbf4f18c81edd3f8780d58b49e18dd6df4a7447c7909e0df', 'parameter_schema_hash': 'dc248551663f556854dad976bc676f7ca5f6abb4117c461f08bd69d3fb97b5a6', 'best_parameter_hash': 'fcf7d158a76f0d6bf59081ecdf99317409142b6d9781d638b99723a1696ace2c', 'best_parameters': {'epsilon': 0.004897482901892843, 'slack_risk_penalty': 2.931895001178948, 'slack_urgency_gain': 3.6542064314476788, 'energy_efficiency_weight': 3.2728371710177537, 'criticality_weight': 1.020827300420228, 'bottleneck_proximity_weight': 2.264146363552534, 'duration_uncertainty_ratio': 0.08002902436585366, 'wait_ramp_threshold': 19.5441996502236, 'smooth_slack_gate_width': 0.0687219804898029, 'finfo_max_scale': 352088804.47115237, 'robust_normalization_quantile': 0.7348207484098567, 'successor_release_sharpness': 1.3899022079527497}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': '3053248efb1c77f3f7f8675cb46ad4670b628917e7c8eff9791849487a5c367b', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule with:
      - Smooth width-controlled sigmoid slack gate (replaces strength-based gate)
      - Monotonic bounded wait ramp (Parent 2)
      - Sharpened successor-release interaction (rank * work * |slack|⁻ˢʰᵃʳᵖⁿᵉˢˢ)
      - Quantile-based robust normalization for all features
      - Uncertainty-slack interaction removed to meet 12-parameter limit while preserving core deadline-energy-criticality balance
      - All numeric literals strictly in {-2,-1,0,1,2}
    """
    eps = 0.004897482901892843
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
        x = np.asarray(x)
        abs_x = np.abs(x)
        finite_mask = np.isfinite(abs_x)
        if np.any(finite_mask):
            scale = np.quantile(abs_x[finite_mask], 0.7348207484098567)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    gate_width = 0.0687219804898029
    ddl_gate = 1.0 / (1.0 + np.exp(-slk / (gate_width + eps)))
    slack_penalty = 2.931895001178948 * np.maximum(-slk, 0.0) + 3.6542064314476788 * np.minimum(slk, 0.0)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -3.2728371710177537 * normalize(inv_energy)
    rank_score = -1.020827300420228 * ddl_gate * normalize(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 1.3899022079527497)
    bottleneck_score = -2.264146363552534 * normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.08002902436585366 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 19.5441996502236)
    wait_normalized = normalize(wait_clipped + eps)
    wait_ramp = np.clip(wait_normalized, 0.0, 1.0)
    wait_score = -wait_ramp
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 352088804.47115237
    min_safe = -finfo.max / 352088804.47115237
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
