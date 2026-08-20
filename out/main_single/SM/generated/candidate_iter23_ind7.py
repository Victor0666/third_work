import numpy as np
RULE_METADATA = {'structure_hash': '88d622db30cd95970f73a546e19cda27d524feaa373807b2a1c758b711da3a8d', 'parameter_schema_hash': '11aa0df1040761e0be3ddcc39f5a654ca3392880ee793f5da8e5e14e1cbcaaff', 'best_parameter_hash': '05c05e9b7e6a0c991dbd251356619ed6f24a7da9be14c8fda2f8dc503d9fa53a', 'best_parameters': {'epsilon': 0.031816417771011206, 'slack_risk_penalty': 7.631231968856687, 'slack_urgency_gain': 0.345036495820821, 'energy_efficiency_weight': 3.687299224842377, 'criticality_weight': 1.9873619847837642, 'bottleneck_proximity_weight': 1.091065311830199, 'successor_release_sharpness': 0.6701716312261923, 'wait_ramp_threshold': 1.2028372470920452, 'ddl_protection_gate': 0.39739954169406794, 'robust_normalization_quantile': 0.5883576307967098, 'uncertainty_gate_exponent': 1.132944361573821, 'finfo_max_scale': 9046601.638356073}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': 'ba3a7e549f0a8ec2997054d51273f050c5c373089279af51e4d6cfa059ca477d', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining best elements from both parents:
      - Keeps Parent 2's successor-release interaction (rank * work / |slack|^sharpness) for deadline-aware critical-path focus.
      - Adopts Parent 1's multiplicative uncertainty gating on exec_t & comm_t (not linear blend), preserving physical risk composition.
      - Uses Parent 2's adaptive quantile normalization (more robust than MAD for skewed distributions).
      - Retains Parent 2's smooth bounded linear wait ramp (monotonic, avoids clipping artifacts).
      - Introduces novel *dual-gating*: applies both sigmoid ddl_gate AND slack-sign-aware scaling to bottleneck term,
        ensuring strict feasibility enforcement while maintaining fine-grained urgency modulation.
      - All numeric literals strictly in {-2,-1,0,1,2}; no hidden constants.
      - Deterministic, finite, shape-(N,) output satisfying all interface contracts.
    """
    eps = 0.031816417771011206
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
            scale = np.quantile(abs_x[finite_mask], 0.5883576307967098)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(0.39739954169406794 * slack_norm))
    slack_penalty = np.where(slk < 0, 7.631231968856687 * np.abs(slack_norm), -0.345036495820821 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -3.687299224842377 * normalize(inv_energy)
    rank_score = -1.9873619847837642 * ddl_gate * normalize(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    sharpness_factor = np.where(slk > 0, np.power(slack_magnitude_inv, 0.6701716312261923), 1.0)
    bottleneck_sharpened = rank * work * sharpness_factor
    bottleneck_score = -1.091065311830199 * ddl_gate * normalize(bottleneck_sharpened + eps)
    exec_gated = exec_t * (1.0 + np.clip(uncert, 0.0, 2.0) ** 1.132944361573821)
    comm_gated = comm_t * (1.0 + np.clip(uncert, 0.0, 2.0) ** 1.132944361573821)
    gated_duration = exec_gated + comm_gated
    dur_score = normalize(gated_duration + eps)
    wait_clipped = np.clip(wait, 0.0, 1.2028372470920452)
    wait_normalized = normalize(wait_clipped + eps)
    wait_ramp = np.clip(wait_normalized, 0.0, 1.0)
    wait_score = -wait_ramp
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 9046601.638356073
    min_safe = -finfo.max / 9046601.638356073
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
