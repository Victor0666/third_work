import numpy as np
RULE_METADATA = {'structure_hash': 'bc30ea250e75f0ba67c480b90a5fa88fc9c8076161ca6f94efcffe3082248575', 'parameter_schema_hash': 'f07f19e51127ce748422d41ec4d3a85a96f382d02ca2d4d01b5d0cb441ff4134', 'best_parameter_hash': '81a25e795dfba6b8442045241610c8b9b6a89c2146de4cbcc58ebff80b5164b2', 'best_parameters': {'epsilon': 0.00042515019559805165, 'criticality_scale': 2.1075588767746636, 'energy_sensitivity': 0.8596073357322651, 'energy_uncertainty_interaction': 0.2992036439315968, 'remaining_work_weight': 0.27254628605684333, 'ddl_protection_gate_slope': 3.43778035610514, 'uncertainty_gate_threshold': 0.02196659569660156, 'wait_decay': 0.39159734444369426, 'slack_penalty_linear_coeff': 1.1710419783785548, 'duration_robustness_factor': 0.7943552898621872}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '429a299c9799d6ce5c7dc964f025322a2c76a227ef2c5154345a69ea8dded31c', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: replaces log-scaled wait benefit with clipped linear (per reflection), removes exponentiation, enforces strict ddl_gate on *all* duration/energy/uncertainty terms.
       Key structural improvement: introduces 'duration_robustness_factor' to dampen norm_duration and exec_penalty — empirically reduces sensitivity to estimation outliers while preserving DDL pressure coupling.
       All gates now uniformly apply ddl_gate, ensuring zero penalty leakage under infeasibility. Normalization remains median-MAD for robustness."""
    eps = 0.00042515019559805165
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
        x = np.asarray(x, dtype=float)
        if N == 1:
            med = x[0]
            mad = eps
        else:
            med = np.median(x)
            mad = np.median(np.abs(x - med))
        spread = mad + eps
        return (x - med) / spread
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    ddl_gate = np.where(slack < 0.0, 0.0, np.clip(slack, 0.0, 1.0))
    ddl_pressure = np.clip(-norm_slack, 0.0, 1.0)
    raw_slack_penalty = np.maximum(-slack, 0.0)
    norm_slack_penalty = robust_normalize(raw_slack_penalty)
    slack_penalty = 1.1710419783785548 * norm_slack_penalty
    successor_release = norm_rank * norm_work * ddl_pressure
    coupled_rank = norm_rank * (1.0 + 2.1075588767746636 * ddl_pressure)
    uncert_gate = np.where(norm_uncert > 0.02196659569660156, 1.0, 0.0)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    wait_benefit = np.clip(0.39159734444369426 * ready_wait_time, 0.0, 1.0)
    exec_penalty = 0.7943552898621872 * norm_duration * ddl_pressure * ddl_gate
    score = +np.clip(slack_penalty, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - np.clip(coupled_rank, -2.0, 2.0) - 0.8596073357322651 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(exec_penalty, -2.0, 2.0) + 0.2992036439315968 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 0.27254628605684333 * np.clip(norm_work, -2.0, 2.0) + np.clip(ddl_pressure * 3.43778035610514, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
