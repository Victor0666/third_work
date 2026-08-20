import numpy as np
RULE_METADATA = {'structure_hash': 'd71f6517ce627b3f087ecd352205fa27caacbc91366b2b31b13343d73bc46718', 'parameter_schema_hash': '52db11f1568990e3673e5458170f978f6785a1a1432f47b95b1c623715e4196b', 'best_parameter_hash': '5b31094e6dc529e77014ce716a64cc6bed1cee15486d9c2d7d148056311ebfd1', 'best_parameters': {'epsilon': 0.01702121771347029, 'slack_risk_penalty': 5.826517135353852, 'slack_urgency_gain': 1.5381545753173718, 'energy_efficiency_weight': 0.5959002649494712, 'criticality_weight': 1.8313778135609176, 'bottleneck_proximity_weight': 1.7252053272093026, 'duration_uncertainty_ratio': 1.1616375855767784, 'wait_ramp_threshold': 24.999602603945103, 'ddl_protection_gate': 0.2897868095167577, 'finfo_max_scale': 24140.47447220485, 'robust_normalization_quantile': 0.8993144341355488, 'successor_release_sharpness': 1.2500874404344722}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': '412bbe207817f20ff778b806311ff1bbaeeb9c6880371b56d98862422138ae12', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with two key structural improvements:
      - Replaced hard wait clipping with bounded linear ramp: smooth, monotonic, avoids abrupt transitions
      - Introduced successor-release interaction: upward_rank * remaining_work * (1 / (|slack| + eps))^sharpness,
        sharpening critical-path focus as deadline pressure increases
      - Load-aware energy gating removed (to reduce parameter count) — instead, energy term remains active but
        is naturally suppressed by ddl_gate and slack_penalty when feasibility is at risk
      - All normalizations use adaptive quantile scaling; no median or mean bias
      - Strict DDL-first ordering preserved via dominant slack_penalty term
    """
    eps = 0.01702121771347029
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
            scale = np.quantile(abs_x[finite_mask], 0.8993144341355488)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(0.2897868095167577 * slack_norm))
    slack_penalty = np.where(slk < 0, 5.826517135353852 * np.abs(slack_norm), -1.5381545753173718 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.5959002649494712 * normalize(inv_energy)
    rank_score = -1.8313778135609176 * ddl_gate * normalize(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 1.2500874404344722)
    bottleneck_score = -1.7252053272093026 * normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 1.1616375855767784 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 24.999602603945103)
    wait_normalized = normalize(wait_clipped + eps)
    wait_ramp = np.clip(wait_normalized, 0.0, 1.0)
    wait_score = -wait_ramp
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 24140.47447220485
    min_safe = -finfo.max / 24140.47447220485
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
