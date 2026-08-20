import numpy as np
RULE_METADATA = {'structure_hash': '2852ad7a48adc4a0d10a0a8a894c7b8a6527a0af8d49b2416d8ef93040acb417', 'parameter_schema_hash': '7dcd905da1b745a070514f5914c6c11db38c6887cc3a5a702b1f06cbd56e2a2a', 'best_parameter_hash': '4d612616945043b3dbffe15291fa8a5facc74d7faa539092b007b2f5b1797617', 'best_parameters': {'epsilon': 0.00025640252898339455, 'slack_penalty_exponent': 2.481130493614286, 'criticality_scale': 1.6816841943491363, 'energy_sensitivity': 0.6780289162369141, 'duration_robustness': 0.5069134034620972, 'wait_decay': 0.09688134363807513, 'uncertainty_gate_threshold': 0.6382939144264805, 'slack_pressure_gate_steepness': 2.536088542212934, 'remaining_work_weight': 0.912987811051038, 'wait_saturation_offset': 8.471681186156678e-05, 'energy_uncertainty_coupling': 0.8390818564651521}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': 'ac7c234745375451f360ed2b65a54e2a85b16a53f4bcf4683d94126e900b1511', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule: all declared parameters used; no unused 'rank_slack_coupling'."""
    eps = 0.00025640252898339455
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
        x = np.abs(x)
        center = np.median(x) if N > 1 else x[0]
        spread = np.median(np.abs(x - center)) if N > 1 else np.abs(x[0] - center) + eps
        return (x - center) / (spread + eps)
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.481130493614286
    norm_slack_penalty = robust_normalize(raw_slack_penalty)
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-2.536088542212934 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * (1.0 + 1.6816841943491363 * rank_gate)
    uncert_gate = np.where(norm_uncert > 0.6382939144264805, 1.0, 0.0)
    duration_risk_score = norm_duration * uncert_gate * slack_pressure
    wait_benefit = 1.0 - np.exp(-0.09688134363807513 * (norm_wait + 8.471681186156678e-05))
    energy_uncertainty_penalty = norm_energy * norm_uncert * 0.8390818564651521
    score = +norm_slack_penalty - boosted_rank - 0.6780289162369141 * norm_energy - norm_duration - wait_benefit + 0.5069134034620972 * duration_risk_score + 0.912987811051038 * norm_work + energy_uncertainty_penalty
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
