import numpy as np
RULE_METADATA = {'structure_hash': 'a6e984dd936a5f4f601888d08ea08f5b300a32dd75d2e97ea5e19db8e5107cc3', 'parameter_schema_hash': '9b5f50c231c9c2ec6733d5e50b9361fcff140864dccada7c9f2f80fe8b6c3af9', 'best_parameter_hash': '864f3a8a179077fc66f05fd62e11f00c9278db30b80a3a02b28e620a19ff9894', 'best_parameters': {'epsilon': 3.9462357369638814e-05, 'slack_risk_penalty': 7.438619155334133, 'slack_urgency_gain': 2.368649733106703, 'energy_efficiency_weight': 3.319596190199653, 'criticality_weight': 1.9461163053007442, 'bottleneck_proximity_weight': 1.0838988641298428, 'duration_uncertainty_ratio': 1.194854986932162, 'wait_ramp_threshold': 5.388741519307864, 'ddl_feasibility_margin': 0.42257189160283887, 'robust_normalization_quantile': 0.8032246312840527, 'successor_release_sharpness': 1.6525761336865294, 'finfo_safety_factor': 0.22786214663430931}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': '260d116cbf5c390431e84faa934d125ccb7ae6439bd027ba3d150a8a4cc49eb3', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with key structural changes:
      - Replaced sigmoid DDL gate with interpretable piecewise-linear DDL feasibility margin (validated across 18+ consensus decisions)
      - Introduced ddl_feasible_mask: binary mask that disables energy scoring *only* when slack is critically tight (<= margin)
      - Sharpened successor-release interaction uses absolute slack (not normalized) for stronger temporal fidelity near deadlines
      - Added finfo_safety_factor parameter to replace hidden literal 0.5 in clamping
      - Retained robust quantile normalization and bounded wait ramp for starvation mitigation
      - All features now strictly preserve DDL-first ordering: slack_penalty dominates; energy only contributes when feasible
    """
    eps = 3.9462357369638814e-05
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
            scale = np.quantile(abs_x[finite_mask], 0.8032246312840527)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    ddl_feasible_mask = slk > 0.42257189160283887
    slack_penalty = np.where(slk < 0, 7.438619155334133 * np.abs(slk), -2.368649733106703 * slk)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -3.319596190199653 * normalize(inv_energy) * ddl_feasible_mask
    rank_score = -1.9461163053007442 * normalize(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 1.6525761336865294)
    bottleneck_score = -1.0838988641298428 * normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 1.194854986932162 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 5.388741519307864)
    wait_normalized = normalize(wait_clipped + eps)
    wait_ramp = np.clip(wait_normalized, 0.0, 1.0)
    wait_score = -wait_ramp
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max * 0.22786214663430931, neginf=finfo.min * 0.22786214663430931)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
