import numpy as np
RULE_METADATA = {'structure_hash': '8d653b7f9e134387d4efd567927381b13e42935f7eb258678b8112c1ceaa2bc1', 'parameter_schema_hash': '0234e08bc4ca88b60c72faea6a7b308a550e5e1ad7c3516030e68e0e210533d5', 'best_parameter_hash': 'fc512026d530c002dc892df772de13425f5ea0b61f3c35983b85ebc1b884e280', 'best_parameters': {'epsilon': 0.001176863893978242, 'slack_risk_penalty': 7.287087984689291, 'slack_urgency_gain': 2.90926481317688, 'energy_efficiency_weight': 3.4043066705743317, 'criticality_weight': 1.3333731058961567, 'bottleneck_proximity_weight': 2.4730749259311344, 'duration_uncertainty_ratio': 0.9896596785275764, 'wait_ramp_threshold': 33.845639055242785, 'ddl_protection_gate': 0.16378457000798652, 'robust_normalization_quantile': 0.8054715636990162, 'host_load_sensitivity': 1.8765765695419785, 'finfo_safe_scale': 1.0283045243201816}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': 'c56df8a0b1e329b95be01bf15b6e3960631c4b0664c7a6c41b53fcd1ff99629e', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with 12 parameters:
      - Removes 'smooth_load_pressure_weight' to comply with 12-parameter limit.
      - Replaces brittle median-based load-pressure clipping with a *robustly normalized* and *fully ddl-gated* version of 'work * uncert',
        but now integrated into the existing bottleneck term via multiplicative coupling — no new parameter needed.
      - Bottleneck term is enhanced: upward_rank * remaining_work * (1 + uncertainty) * slack_magnitude_inv, all quantile-normalized.
      - This preserves structural expressivity while eliminating unbounded interactions and reducing parameter count.
      - All numeric literals are -2, -1, 0, 1, or 2; epsilon and finfo handled via PARAMS and np.finfo.
      - Strictly enforces finite output, shape (N,), and deterministic behavior.
    """
    eps = 0.001176863893978242
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
            scale = np.quantile(abs_x[finite_mask], 0.8054715636990162)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(0.16378457000798652 * slack_norm))
    slack_penalty = np.where(slk < 0, 7.287087984689291 * np.abs(slack_norm), -2.90926481317688 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -3.4043066705743317 * normalize(inv_energy)
    rank_score = -1.3333731058961567 * ddl_gate * normalize(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_base = rank * work * (1.0 + uncert) * slack_magnitude_inv
    bottleneck_sharpened = np.log1p(np.exp(bottleneck_base))
    bottleneck_score = -2.4730749259311344 * normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.9896596785275764 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_normalized = wait / (33.845639055242785 + eps)
    wait_saturation = 1.0 - np.exp(-wait_normalized)
    wait_saturation = np.clip(wait_saturation, 0.0, 1.0)
    wait_score = -wait_saturation
    energy_per_duration = energy / (duration + eps)
    load_proxy = normalize(energy_per_duration + eps)
    load_score = -1.8765765695419785 * ddl_gate * load_proxy
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score + load_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 1.0283045243201816
    min_safe = finfo.min / 1.0283045243201816
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
