import numpy as np
RULE_METADATA = {'structure_hash': '8ff9de0d60552288be23ad9d2ae962d4a3caae18caf0a173770b21f3c35f8e5e', 'parameter_schema_hash': '0be94cb6b1a988f0ecf28ae38c92c20de8906ccf0a01cf142b702a480f9b359d', 'best_parameter_hash': '9a93118b342574a77e335b717ee16223bcb9d3e0c4d472a6578df9c59f153c53', 'best_parameters': {'epsilon': 0.035867074120047394, 'slack_penalty_exponent': 1.023903833465822, 'criticality_scale': 1.6824909872411211, 'energy_sensitivity': 0.46270479945672316, 'duration_robustness': 0.7162135845165029, 'wait_decay': 0.35192172288976376, 'uncertainty_gate_threshold': 1.058340283052139e-06, 'slack_gate_width': 0.8752120545110816, 'energy_slack_interaction': 1.1613412348839331, 'rank_slack_coupling': 0.5670217412410496, 'uncertainty_smoothness': 2.337890754244037, 'lateness_penalty_strength': 1.2365910473287474}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '7de4f24c003ce794730fb73a6be697d88df8ee20627fb246bbad407cc576170f', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: adds hard-deadline enforcement via lateness penalty multiplier.
    Key improvements:
      - Introduces `lateness_penalty_strength`: multiplies *all* penalty terms (not rewards) when slack is critically negative,
        ensuring strict adherence to hard deadlines without distorting intra-feasible prioritization.
      - Reverts to original robust_normalize (MAD-only, no clipping) per reflection — preserves rank fidelity under stress.
      - Restores `slack_pressure` (not raw) in energy/rank terms for stronger nonlinear urgency amplification.
      - Removes successor pressure term as instructed; avoids unbounded upstream propagation.
      - All numeric literals remain in {-2,-1,0,1,2}; uses np.finfo for safe finite bounds.
    """
    eps = 0.035867074120047394
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def robust_normalize(x):
        x_abs = np.abs(x)
        center = np.median(x_abs)
        scale = np.median(np.abs(x_abs - center)) + eps
        return (x_abs - center) / scale
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    slack_pressure_raw = np.clip(-norm_slack, 0.0, None)
    slack_pressure = np.power(slack_pressure_raw + eps, 1.023903833465822)
    is_late = (slack < -eps).astype(float)
    lateness_multiplier = 1.0 + 1.2365910473287474 * is_late
    rank_gate = np.clip(1.0 - norm_slack / (0.8752120545110816 + eps), 0.0, 1.0)
    boosted_rank = norm_rank * (1.0 + 1.6824909872411211 * rank_gate)
    uncertainty_activation = 1.0 / (1.0 + np.exp(-2.337890754244037 * (norm_uncert - 1.058340283052139e-06)))
    duration_risk_interaction = norm_duration * uncertainty_activation * 0.7162135845165029
    wait_benefit = 1.0 - np.exp(-0.35192172288976376 * (norm_wait + eps))
    energy_slack_penalty = norm_energy * (1.0 + 1.1613412348839331 * slack_pressure)
    coupled_rank_reward = norm_rank * (1.0 + 0.5670217412410496 * slack_pressure)
    score = +lateness_multiplier * slack_pressure + lateness_multiplier * energy_slack_penalty + lateness_multiplier * 0.46270479945672316 * norm_energy - coupled_rank_reward - wait_benefit + lateness_multiplier * duration_risk_interaction
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=finfo.max, posinf=finfo.max, neginf=finfo.min)
    assert score.shape == (N,), f'Expected shape {(N,)}, got {score.shape}'
    return score
