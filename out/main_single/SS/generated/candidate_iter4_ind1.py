import numpy as np
RULE_METADATA = {'structure_hash': '8bb33f208844b04f45a649b6a635666cd0ba9baa4e601593be6c80904d2f135f', 'parameter_schema_hash': '63c95399b43307d8b84bc0fe44c0d1e87c9cd963b731ee35f83c0b64ab1f8322', 'best_parameter_hash': '26ef90c6c0fe5cc8524ad1e641856eb44dc0eeb92047d6bf424f4f6233bad30c', 'best_parameters': {'epsilon': 0.018960355631403154, 'slack_penalty_exponent': 2.7797401247441402, 'criticality_scale': 1.2690636606909824, 'energy_sensitivity': 1.9966170311366427, 'duration_robustness': 0.5190255769932859, 'wait_decay': 0.3106375034861657, 'uncertainty_gate_threshold': 0.48221439632172486, 'slack_pressure_gate_steepness': 2.8768534557757484, 'remaining_work_weight': 0.6885512325161512, 'wait_saturation_offset': 7.146307389676298e-08, 'energy_uncertainty_interaction': 0.7599006561307149, 'ddl_protection_gate_threshold': 0.5735338644214149}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': 'd174d444e32c43e4a7b3150f23ee0e0b48485735d84c38dd4287c80bfcc6abf3', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule: replaces robust_normalize with mean-abs scaling for stability;
       introduces conditional DDL-protection gate; replaces exponential wait relief with tanh-based anti-starvation;
       adds successor-release interaction via remaining_work * slack coupling under deadline pressure."""
    eps = 0.018960355631403154
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def stable_normalize(x):
        x_abs = np.abs(x)
        scale = np.mean(x_abs) + eps
        return x / (scale + eps)
    norm_slack = stable_normalize(slack)
    norm_energy = stable_normalize(min_incremental_energy)
    norm_duration = stable_normalize(min_exec_time + min_comm_time)
    norm_rank = stable_normalize(upward_rank)
    norm_work = stable_normalize(remaining_work)
    norm_wait = stable_normalize(ready_wait_time)
    norm_uncert = stable_normalize(uncertainty)
    slack_pressure = np.tanh(-norm_slack * 2.8768534557757484)
    ddl_protection_gate = np.clip(slack_pressure, 0.0, 1.0)
    boosted_rank = norm_rank * (1.0 + 1.2690636606909824 * ddl_protection_gate)
    successor_release_bonus = norm_work * ddl_protection_gate
    wait_benefit = np.tanh(0.3106375034861657 * (norm_wait + 7.146307389676298e-08))
    uncert_gate = (norm_uncert > 0.48221439632172486) & (ddl_protection_gate > 0.5735338644214149)
    duration_risk_score = norm_duration * uncert_gate.astype(float) * ddl_protection_gate
    energy_uncert_penalty = norm_energy * norm_uncert * ddl_protection_gate * uncert_gate.astype(float)
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.7797401247441402
    norm_slack_penalty = stable_normalize(raw_slack_penalty)
    score = +norm_slack_penalty - boosted_rank - 1.9966170311366427 * norm_energy - norm_duration - wait_benefit + 0.5190255769932859 * duration_risk_score + 0.7599006561307149 * energy_uncert_penalty - 0.6885512325161512 * successor_release_bonus
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
