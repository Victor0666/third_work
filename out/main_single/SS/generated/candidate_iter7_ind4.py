import numpy as np
RULE_METADATA = {'structure_hash': '5c50914dc27d40550c6db25e3a2372197c3e96a94cba24480d8928022fdcc8c3', 'parameter_schema_hash': 'd93ce43271c0941729af5953bab362838337a693c3343f5b97b4c9773ed2e556', 'best_parameter_hash': 'e6a77e4671db2e7affe8c34be2420d8b6f4fc8c1eecde35c4b36804b708e2eb4', 'best_parameters': {'epsilon': 0.004990712857806658, 'slack_penalty_exponent': 1.0313853815832434, 'criticality_scale': 1.2250225171114328, 'energy_sensitivity': 0.48921755642696807, 'duration_robustness': 0.20749480017358235, 'wait_decay': 0.08708662871555123, 'uncertainty_gate_threshold': 0.7033338429012551, 'slack_pressure_gate_steepness': 3.8184834453062653, 'remaining_work_weight': 1.2532092173889309, 'wait_saturation_offset': 4.127854461845252e-09, 'energy_uncertainty_interaction': 0.8895380002697731, 'ddl_protection_gate_strength': 1.1508829040425637}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '777d2d3c912d88fced97e9bbeebeb59df830e1ca913be13ee029b517d4c287d3', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule: 12 parameters, all used; no unused 'successor_release_weight'."""
    eps = 0.004990712857806658
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
    ddl_protection_gate = 1.0 / (1.0 + np.exp(-1.1508829040425637 * (slack_distance + eps)))
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.0313853815832434
    norm_slack_penalty = robust_normalize(raw_slack_penalty)
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-3.8184834453062653 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * (1.0 + 1.2250225171114328 * rank_gate)
    uncert_gate = np.where(norm_uncert > 0.7033338429012551, 1.0, 0.0)
    duration_risk_score = norm_duration * uncert_gate * ddl_protection_gate
    wait_benefit = 1.0 - np.exp(-0.08708662871555123 * (norm_wait + 4.127854461845252e-09))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate
    score = +norm_slack_penalty - boosted_rank - 0.48921755642696807 * norm_energy - norm_duration - wait_benefit + 0.20749480017358235 * duration_risk_score + 0.8895380002697731 * energy_uncert_penalty + 1.2532092173889309 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
