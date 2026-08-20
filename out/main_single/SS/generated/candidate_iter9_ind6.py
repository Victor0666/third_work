import numpy as np
RULE_METADATA = {'structure_hash': '0c9820df44dea30e141aa64e6891381d77ce69350b49db046a53d1a0705d0789', 'parameter_schema_hash': '5b3d33a88c3f20637b3ffddbba47496360a3a428f012a2ae43156bb237d7bb83', 'best_parameter_hash': '536bbe3a5ab486c2ea9dfca00c154c439845ac4cc2b81e4d644156377e3bcbc8', 'best_parameters': {'epsilon': 0.03974768170292612, 'slack_penalty_exponent': 1.7827650046439305, 'criticality_scale': 1.0190129050886294, 'energy_sensitivity': 1.691674602161666, 'duration_robustness': 0.9458681333254253, 'wait_decay': 0.20352165800418176, 'uncertainty_gate_threshold': 0.8475079582028255, 'remaining_work_weight': 1.0735480132719772, 'wait_saturation_offset': 6.252759851650153e-08, 'energy_uncertainty_interaction': 0.03312822835781615, 'ddl_urgency_steepness': 4.38529421055534, 'rank_boost_floor': 0.06073062612333798}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '430aeed23c013aceae48835cf7dd7cd8bd7aae014b56924afff4635843136579', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: replaces fragile binary gate with smooth sigmoid DDL urgency gate,
       adds rank_boost_floor to prevent critical-path suppression under mild pressure,
       reverts to pure raw-slack-penalty dominance for violation enforcement,
       and simplifies interactions to improve robustness and CMA-ES convergence.
       All 12 parameters are used; no numeric literals beyond -2,-1,0,1,2."""
    eps = 0.03974768170292612
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
    urgency_input = -slack / (np.abs(np.median(slack)) + eps) if N > 1 else -slack[0] / (np.abs(slack[0]) + eps)
    ddl_urgency_gate = 1.0 / (1.0 + np.exp(-4.38529421055534 * urgency_input))
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.7827650046439305
    rank_boost = 0.06073062612333798 + (1.0 - 0.06073062612333798) * ddl_urgency_gate
    protected_rank = norm_rank * rank_boost
    wait_benefit = np.clip(0.20352165800418176 * (ready_wait_time + 6.252759851650153e-08), 0.0, 1.0)
    energy_uncert_penalty = norm_energy * norm_uncert * np.where((norm_energy > 0.0) & (norm_uncert > 0.8475079582028255), 1.0, 0.0)
    duration_risk_score = norm_duration * np.where((slack <= 0.03974768170292612) & (uncertainty >= 0.8475079582028255), 1.0, 0.0)
    score = +raw_slack_penalty - 1.0190129050886294 * protected_rank - 1.691674602161666 * norm_energy - norm_duration - wait_benefit + 0.9458681333254253 * duration_risk_score + 0.03312822835781615 * energy_uncert_penalty + 1.0735480132719772 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
