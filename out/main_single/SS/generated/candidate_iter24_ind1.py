import numpy as np
RULE_METADATA = {'structure_hash': '04accf336ec95d5a2cb02ee2c42d8319c90c8cbae491136b83a257d74d819baa', 'parameter_schema_hash': 'cd0fc84e32772ac94bb061751b60dfb7052fb92d5f634702229c5e41c779fa53', 'best_parameter_hash': '6b91e91d90e5dc0500d802303b4b78e48f46dcc5e7624fcb59257bd08486bd0d', 'best_parameters': {'epsilon': 6.960561873495651e-05, 'criticality_scale': 1.8239057010731223, 'energy_sensitivity': 0.383337077925509, 'energy_uncertainty_interaction': 0.41827932905939585, 'remaining_work_weight': 1.7187821149889695, 'uncertainty_gate_threshold': 0.030620845480427077, 'wait_decay': 0.12664081532637012, 'slack_penalty_linear_coeff': 2.2238884000514334, 'duration_robustness_factor': 1.1841895134827147, 'slack_energy_suppression_strength': 0.9275882623762294, 'host_load_sensitivity': 0.9323401413396517}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '1f60ca0547173c9c7d39386c914fc6c3b04b09aa61eacd7e16e019d6f9cc7414', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: replaces unstable ddl_pressure² with smooth, bounded inverse-pressure 
       using finite-floor |slack|+eps; restores additive critical-path coupling (rank × work × pressure);
       removes unused ddl_protection_gate_slope; retains all safety gates, robust normalization, and feasibility-first logic."""
    eps = 6.960561873495651e-05
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
    hard_feasibility_gate = np.where(slack < 0.0, 0.0, 1.0)
    slack_abs = np.abs(slack) + eps
    ddl_pressure = np.clip(1.0 / (1.0 + slack_abs), 0.0, 1.0)
    raw_slack_penalty = np.maximum(-slack, 0.0)
    norm_slack_penalty = robust_normalize(raw_slack_penalty)
    slack_penalty = 2.2238884000514334 * norm_slack_penalty
    successor_release = norm_rank * norm_work * ddl_pressure
    coupled_rank = norm_rank * (1.0 + 1.8239057010731223 * ddl_pressure)
    uncert_gate = np.where(norm_uncert > 0.030620845480427077, 1.0, 0.0)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * hard_feasibility_gate
    wait_benefit = np.clip(0.12664081532637012 * ready_wait_time, 0.0, 2.0)
    duration_penalty = 1.1841895134827147 * norm_duration * ddl_pressure * hard_feasibility_gate
    slack_energy_suppress = np.where(slack < 0.0, 1.0 - 0.9275882623762294, 1.0)
    suppressed_norm_energy = norm_energy * slack_energy_suppress
    load_surrogate = norm_duration * norm_uncert
    load_gate = np.where((hard_feasibility_gate > 0.0) & (norm_uncert > 0.030620845480427077), np.clip(load_surrogate, 0.0, 2.0), 0.0)
    host_load_penalty = 0.9323401413396517 * load_gate * suppressed_norm_energy
    score = +np.clip(slack_penalty, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - np.clip(coupled_rank, -2.0, 2.0) - 0.383337077925509 * np.clip(suppressed_norm_energy, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(duration_penalty, -2.0, 2.0) + 0.41827932905939585 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 1.7187821149889695 * np.clip(norm_work, -2.0, 2.0) + host_load_penalty
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.reshape(-1)
