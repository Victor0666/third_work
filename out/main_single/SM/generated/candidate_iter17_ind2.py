import numpy as np
RULE_METADATA = {'structure_hash': 'd8c181f79dca5a7858ee7ad83a067bbc3b8a2e9c516b1818d386729e532c098f', 'parameter_schema_hash': 'ad22a7a9f5c17d6f7a0a6e611663fd04017b53ed40b97dd035eaa3bd0048943f', 'best_parameter_hash': '360771076ebbfc5cfce01f081a917bf32dd948c7c00723eb957ae3a5971a46cc', 'best_parameters': {'epsilon': 0.00030426112380933795, 'slack_risk_penalty': 2.684219741209874, 'slack_urgency_gain': 1.4796822715015066, 'energy_efficiency_weight': 1.032659062940632, 'criticality_weight': 1.46878655264574, 'bottleneck_proximity_weight': 2.6449952111211488, 'duration_uncertainty_ratio': 1.19262155645748, 'wait_ramp_threshold': 3.869824939405847, 'ddl_protection_gate': 0.00808331414123237, 'finfo_max_scale': 22627260.826697696, 'robust_normalization_quantile': 0.841646660312815, 'successor_release_sharpness': 1.6405004258772538}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': 'd564b3f4ee54d39b921b4350db5f90000d8021497a64342031394348c53ce22c', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule with:
      - Smooth sigmoid ddl_gate + sharpened successor-release term
      - Linear wait saturation (exponent=1, literal allowed)
      - Multiplicative energy-slack coupling via ddl_gate^1
      - Robust quantile normalization throughout
      - All numeric literals restricted to {-2,-1,0,1,2}
      - Exactly 12 parameters
    """
    eps = 0.00030426112380933795
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
            scale = np.quantile(abs_x[finite_mask], 0.841646660312815)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(0.00808331414123237 * slack_norm))
    slack_penalty = np.where(slk < 0, 2.684219741209874 * np.abs(slack_norm), -1.4796822715015066 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.032659062940632 * normalize(inv_energy)
    energy_score = energy_score * np.power(ddl_gate, 1)
    rank_score = -1.46878655264574 * ddl_gate * normalize(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 1.6405004258772538)
    bottleneck_score = -2.6449952111211488 * normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 1.19262155645748 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 3.869824939405847)
    wait_normalized = normalize(wait_clipped + eps)
    wait_saturation = np.clip(wait_normalized, 0.0, 1.0)
    wait_score = -wait_saturation
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 22627260.826697696
    min_safe = -finfo.max / 22627260.826697696
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
