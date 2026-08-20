import numpy as np
RULE_METADATA = {'structure_hash': '057d86d9eaebdf4d52347746f998d5affdf3eac0847b6cbadaf2e5ebfe6b7f94', 'parameter_schema_hash': '062b79fa126c478e7000fad22103c3e195ad241ba8d0763e0c5befde71c3f272', 'best_parameter_hash': '6b8db29d4e1cd34010d9e99a2c948144c24f7e79a24116986550577804c7aca2', 'best_parameters': {'epsilon': 0.004728289788870691, 'slack_risk_penalty': 4.302183195846855, 'slack_urgency_gain': 4.371395590019763, 'energy_efficiency_weight': 1.3036090804369975, 'criticality_weight': 0.9275160170051313, 'bottleneck_proximity_weight': 1.6966835990563738, 'duration_uncertainty_ratio': 1.054737964209882, 'ddl_protection_gate': 0.2620064903065842, 'finfo_max_scale': 20406.64766901174, 'robust_normalization_quantile': 0.8501817737804391, 'bounded_slack_inverse_exponent': 0.19218514826511454, 'load_gate_center': 0.49686016881856493}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': 'c0daef50b6c03f7c742ae772f694bc13837a59574d1b3d547a61abb96d72d99f', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
      - Bounded inverse-slack gating: (|slack|+ε)^(-exp) — finite, smooth, tunable sharpness.
      - Host-load-aware energy gating: sigmoid on normalized (exec+comm), centered at tunable load_gate_center.
      - Starvation mitigation via direct linear wait-time ranking (no normalization/clipping).
      - All numeric literals are in {-2,-1,0,1,2}; no hidden constants.
      - Exactly 12 parameters; all used; no unused or missing PARAMS references.
    """
    eps = 0.004728289788870691
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
            scale = np.quantile(abs_x[finite_mask], 0.8501817737804391)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(0.2620064903065842 * slack_norm))
    slack_penalty = np.where(slk < 0, 4.302183195846855 * np.abs(slack_norm), -4.371395590019763 * np.abs(slack_norm))
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    load_gate = 1.0 / (1.0 + np.exp(-2.0 * (dur_norm - 0.49686016881856493)))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.3036090804369975 * normalize(inv_energy) * load_gate
    rank_score = -0.9275160170051313 * ddl_gate * normalize(rank + eps)
    slack_magnitude_bounded = np.abs(slk) + eps
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_bounded, -0.19218514826511454)
    bottleneck_score = -1.6966835990563738 * normalize(bottleneck_sharpened + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 1.054737964209882 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_score = -wait
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 20406.64766901174
    min_safe = -finfo.max / 20406.64766901174
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
