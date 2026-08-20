import numpy as np
RULE_METADATA = {'structure_hash': '21c373538002681d8a453c440d2f9769cf3eab667901b211845e0f59c44050cf', 'parameter_schema_hash': '70c55972e6f9368405b13c78069611d3cefe71547a94cc5cec38f6af46d68af5', 'best_parameter_hash': '9c93b3c1b01b578f9b4e7a1b783232907a3763ad671944caf3935761e85534d3', 'best_parameters': {'epsilon': 0.09458645060231904, 'slack_penalty_exponent': 3.018690356769438, 'criticality_scale': 0.5062651450496246, 'energy_sensitivity': 1.6771302730830768, 'duration_robustness': 0.5913604841113299, 'wait_decay': 0.02563700162242031, 'uncertainty_gate_threshold': 0.6198552883364121, 'slack_pressure_gate_steepness': 4.125940760197873, 'remaining_work_weight': 1.1161381591065855, 'wait_saturation_offset': 4.2222190150742933e-08, 'energy_uncertainty_interaction': 0.7775153285196758}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '89e181b7c62f5b13d7b4179aeae1a673a32acccf118165c3b094e10525c922a7', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule: replaces linear rank coupling with successor-release interaction;
       adds conditional DDL protection gate; uses median-MAD per-feature normalization;
       enforces hard deadline feasibility via binary urgency gating before composite scoring."""
    eps = 0.09458645060231904
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
    ddl_urgent_mask = (slack <= 0.0).astype(float)
    successor_release_score = norm_rank * norm_work * ddl_urgent_mask
    slack_gap = np.maximum(-slack, 0.0)
    ddl_protection_gate = 1.0 / (1.0 + np.exp(-4.125940760197873 * (slack_gap - eps)))
    boosted_rank = norm_rank * (1.0 + 0.5062651450496246 * ddl_protection_gate)
    energy_uncert_penalty = norm_energy * norm_uncert * np.where(norm_uncert > 0.6198552883364121, 1.0, 0.0)
    duration_risk_score = norm_duration * norm_uncert * ddl_protection_gate
    wait_benefit = 1.0 - np.exp(-0.02563700162242031 * (norm_wait + 4.2222190150742933e-08))
    score = +robust_normalize(np.maximum(-slack, 0.0) ** 3.018690356769438) - successor_release_score - boosted_rank - 1.6771302730830768 * norm_energy - wait_benefit + 0.5913604841113299 * duration_risk_score + 0.7775153285196758 * energy_uncert_penalty + 1.1161381591065855 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
