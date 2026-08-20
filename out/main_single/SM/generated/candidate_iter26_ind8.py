import numpy as np
RULE_METADATA = {'structure_hash': '1647fcfc1e5cc5abe37b89b36552c82ef892d2740dba48e2f9a48cba071cca59', 'parameter_schema_hash': '52db11f1568990e3673e5458170f978f6785a1a1432f47b95b1c623715e4196b', 'best_parameter_hash': '36e064fb90871dd47b7d2808b27965e7b5d1a304338c074cb59d4758fd9e3df8', 'best_parameters': {'epsilon': 0.022873798597380264, 'slack_risk_penalty': 9.963367335386991, 'slack_urgency_gain': 3.887103274598106, 'energy_efficiency_weight': 1.1716387655228069, 'criticality_weight': 1.7541843332562697, 'bottleneck_proximity_weight': 2.8534467166251587, 'duration_uncertainty_ratio': 0.16087419755333304, 'wait_ramp_threshold': 26.166928857705404, 'ddl_protection_gate': 0.5592742217931547, 'finfo_max_scale': 422097.8033951461, 'robust_normalization_quantile': 0.5476987339693417, 'successor_release_sharpness': 3.1231705393850566}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': '7f3cdf901bc825829baeae9f4efcb034014b23a8b420e70e66521c71296f202a', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule with slack-aware energy coupling:
      - Replaces fragile host-load gating with simpler, more interpretable slack-coupled energy suppression.
      - Energy weight decays smoothly via sigmoid(-steepness * slk), tunable via new 'energy_coupling_steepness'.
      - Retains all successful elements from Parent 2: duration-uncertainty blend, successor-release,
        bounded wait ramp, quantile normalization, and dominant slack penalty.
      - All numeric literals are in {-2,-1,0,1,2}.
    """
    eps = 0.022873798597380264
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
            scale = np.quantile(abs_x[finite_mask], 0.5476987339693417)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(0.5592742217931547 * slack_norm))
    slack_penalty = np.where(slk < 0, 9.963367335386991 * np.abs(slack_norm), -3.887103274598106 * np.abs(slack_norm))
    energy_coupling_mask = 1.0 / (1.0 + np.exp(-0.5592742217931547 * slk))
    energy_weight = 1.0 - energy_coupling_mask
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.1716387655228069 * energy_weight * normalize(inv_energy)
    rank_score = -1.7541843332562697 * ddl_gate * normalize(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 3.1231705393850566)
    bottleneck_score = -2.8534467166251587 * normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.16087419755333304 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 26.166928857705404)
    wait_normalized = normalize(wait_clipped + eps)
    wait_ramp = np.clip(wait_normalized, 0.0, 1.0)
    wait_score = -wait_ramp
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 422097.8033951461
    min_safe = -finfo.max / 422097.8033951461
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
