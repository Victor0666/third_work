import numpy as np
RULE_METADATA = {'structure_hash': '616d95bcca573dd669a0dc5a9b46373813100990f23ab1d97699a7372903bd9c', 'parameter_schema_hash': 'c34fbcf897f7f97892a4716cbca55cbf4b8898d1d656b9423f4d668031246706', 'best_parameter_hash': 'ff9357a718d3d1a0df7ab98e5cbad7389e4c866e4c4b792fd1eae78d6a083ae0', 'best_parameters': {'epsilon': 4.476708973794298e-06, 'slack_penalty_exponent': 2.8233787336449407, 'criticality_scale': 0.8503723630757377, 'energy_sensitivity': 1.2775078750429683, 'duration_robustness': 1.1082009362667296, 'wait_decay': 0.31238693105756177, 'uncertainty_gate_threshold': 0.9684550094290425, 'remaining_work_weight': 0.5472677976792457, 'wait_saturation_offset': 2.1955918639037865e-09, 'energy_uncertainty_interaction': 0.6721249685640308}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '6e8849fdea05595b4328d6b04e9528b0feaac4779a572ef6e30303ee97e5a963', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule: removes unused 'slack_pressure_gate_steepness'; uses binary DDL protection gate,
       successor-release interaction, and simplified robust normalization. All declared parameters are used."""
    eps = 4.476708973794298e-06
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
    ddl_urgent = np.where((slack <= 0.0) | (slack <= 4.476708973794298e-06) & (uncertainty >= 0.9684550094290425), 1.0, 0.0)
    protected_rank = norm_rank * (1.0 + 0.8503723630757377 * ddl_urgent)
    successor_release_score = norm_work * norm_rank
    energy_uncert_penalty = norm_energy * norm_uncert * np.where((norm_energy > 0.0) & (norm_uncert > 0.9684550094290425), 1.0, 0.0)
    wait_benefit = np.clip(0.31238693105756177 * (ready_wait_time + 2.1955918639037865e-09), 0.0, 1.0)
    duration_risk_score = norm_duration * np.where((norm_uncert > 0.9684550094290425) & (slack <= 0.0), 1.0, 0.0)
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.8233787336449407
    norm_slack_penalty = robust_normalize(raw_slack_penalty)
    score = +norm_slack_penalty - protected_rank - successor_release_score - 1.2775078750429683 * norm_energy - norm_duration + 1.1082009362667296 * duration_risk_score + 0.6721249685640308 * energy_uncert_penalty - wait_benefit + 0.5472677976792457 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
