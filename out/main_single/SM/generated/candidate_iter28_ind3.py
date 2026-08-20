import numpy as np
RULE_METADATA = {'structure_hash': 'fb75416ec7f83947e706f36897427de5872701135cdcb98d92612ae027695eda', 'parameter_schema_hash': 'f76352ff973db0bde9cd07eb07f8ab070b94120f7b4b8fb0cfc60c34f77f36b0', 'best_parameter_hash': '67a918ee9f2bae2ee81909b64486e248a728467f4c4d3fff199ad8e9152a79c8', 'best_parameters': {'epsilon': 2.6776105484284104e-05, 'slack_risk_penalty': 2.80793831515969, 'slack_urgency_gain': 1.9154757697780176, 'energy_efficiency_weight': 2.293696411895781, 'criticality_weight': 0.4258320829818744, 'bottleneck_proximity_weight': 0.9277787397621432, 'duration_uncertainty_ratio': 1.6629562587294202, 'ddl_protection_gate': 0.19309295544898541, 'robust_normalization_quantile': 0.9471510949485189, 'successor_release_sharpness': 0.896333000113992, 'host_load_suppression_factor': 0.3112519096413835, 'piecewise_linear_midpoint': 0.718566397689471}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': 'cc15edc5fec0920ad999ae43b1726d927539f0f75a5fd4939bc430ab92d4085f', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with three structural improvements:
      - Added host-load-aware energy suppression: energy optimization only activated when DDL safety margin exceeds threshold
      - Replaced linear wait ramp with smooth power-law saturation: preserves monotonicity while reducing sensitivity to outliers
      - Introduced piecewise-linear robust slack gating: replaces sigmoid with interpretable, bounded transition around slack=0
      - Removed finfo_max_scale (inactive per diagnostics) and simplified normalization safeguards
      - All features now strictly normalized using quantile scaling with epsilon fallback
      - Critical-path terms retain dominance via slack penalty + bottleneck interaction
    """
    eps = 2.6776105484284104e-05
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
            scale = np.quantile(abs_x[finite_mask], 0.9471510949485189)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    ddl_gate_width = 0.19309295544898541
    midpoint = 0.718566397689471
    ddl_gate = np.clip(midpoint + midpoint * (slk / (ddl_gate_width + eps)), 0.0, 1.0)
    slack_penalty = np.where(slk < 0, 2.80793831515969 * np.abs(slk), -1.9154757697780176 * slk)
    safe_margin = slk - (exec_t + comm_t)
    energy_active = (safe_margin > eps).astype(np.float64)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -2.293696411895781 * normalize(inv_energy) * energy_active
    load_proxy = normalize(uncert + eps) + normalize(exec_t + eps)
    load_suppression = 1.0 - 0.3112519096413835 * np.clip(load_proxy, 0.0, 1.0)
    energy_score = energy_score * load_suppression
    rank_score = -0.4258320829818744 * ddl_gate * normalize(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 0.896333000113992)
    bottleneck_score = -0.9277787397621432 * normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 1.6629562587294202 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_normalized = normalize(wait + eps)
    wait_saturation = np.power(np.clip(wait_normalized, 0.0, 1.0), 0.3112519096413835)
    wait_score = -wait_saturation
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
