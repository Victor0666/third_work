import numpy as np
RULE_METADATA = {'structure_hash': 'e52178354f6833727d5fed40fc3f8733af6c8abed94ecaf96eff7cad24816c11', 'parameter_schema_hash': 'd93ce43271c0941729af5953bab362838337a693c3343f5b97b4c9773ed2e556', 'best_parameter_hash': '65b9d1230c6d6829337cac972a03cfe390743feea4c7c8cf53172d4f4698e57b', 'best_parameters': {'epsilon': 0.000979864429795473, 'slack_penalty_exponent': 2.763872476001701, 'criticality_scale': 1.5097053849481437, 'energy_sensitivity': 1.5987185181331212, 'duration_robustness': 0.6164527842205896, 'wait_decay': 0.28873199697766716, 'uncertainty_gate_threshold': 0.4743374233826809, 'slack_pressure_gate_steepness': 1.059367137708898, 'remaining_work_weight': 0.14048226430031266, 'wait_saturation_offset': 1.1895213663560852e-05, 'energy_uncertainty_interaction': 0.00792561093498291, 'ddl_protection_gate_strength': 0.26003544600790207}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '0bede250ce99a8c33f945b59db2d57c3c351d7f75dd95a3e4c343a74ccaa1b21', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule: adds conditional DDL protection gate; removes redundant successor_release_weight."""
    eps = 0.000979864429795473
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
        center = np.median(x_abs) if N > 1 else x_abs[0]
        spread = np.median(np.abs(x_abs - center)) if N > 1 else np.abs(x_abs[0] - center) + eps
        return (x_abs - center) / (spread + eps)
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    slack_distance = -slack
    ddl_protection_gate = 1.0 / (1.0 + np.exp(-0.26003544600790207 * (slack_distance + eps)))
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.763872476001701
    norm_slack_penalty = robust_normalize(raw_slack_penalty)
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-1.059367137708898 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * (1.0 + 1.5097053849481437 * rank_gate)
    uncert_gate = np.where(norm_uncert > 0.4743374233826809, 1.0, 0.0)
    duration_risk_score = norm_duration * uncert_gate * ddl_protection_gate
    wait_benefit = 1.0 - np.exp(-0.28873199697766716 * (norm_wait + 1.1895213663560852e-05))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate
    score = +norm_slack_penalty - boosted_rank - 1.5987185181331212 * norm_energy - norm_duration - wait_benefit + 0.6164527842205896 * duration_risk_score + 0.00792561093498291 * energy_uncert_penalty + 0.14048226430031266 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
