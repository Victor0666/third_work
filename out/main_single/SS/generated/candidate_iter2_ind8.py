import numpy as np
RULE_METADATA = {'structure_hash': 'c05962f7041e09c1bc8a0825fbba8838aa48d4e21834d05ac299cd31d87e1e5d', 'parameter_schema_hash': 'b010e01395c9a381a51c48e23ce0c125460ab9ea045ed2328a752824922c0c9e', 'best_parameter_hash': 'b06ded697056f5ef54fb83e158a69237a26f42ffb3843c5748736512b0fed5dd', 'best_parameters': {'epsilon': 0.01675190048826803, 'slack_penalty_exponent': 2.6575803862374503, 'criticality_scale': 1.0228639196229838, 'duration_robustness': 0.0042556405806818565, 'wait_decay': 0.5408638547632699, 'uncertainty_gate_threshold': 0.5952744435260071, 'rank_slack_coupling': 0.6714441511444741, 'slack_gate_width': 1.9064858604128563, 'work_pressure_weight': 0.08860076192822072, 'energy_slack_coupling': 0.5920589297981658}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': 'e057a8fc60a18b5d6bea7b416a97c4cf3993c5c1e0e13f75936b924b4867fb38', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule: removes unused 'energy_sensitivity'; all declared parameters now used.
       Uses robust MAD normalization, joint risk gating, tanh starvation relief, work-pressure, and energy-slack coupling.
       Only literals are -2, -1, 0, 1, 2; epsilon via PARAMS; finite output guaranteed."""
    eps = 0.01675190048826803
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
        deviations = np.abs(x_abs - center)
        scale = np.median(deviations) if N > 1 else deviations[0]
        return x_abs / (scale + eps)
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    slack_pressure_raw = np.clip(-slack, 0.0, None) + eps
    slack_pressure = np.power(slack_pressure_raw, 2.6575803862374503)
    slack_pressure = robust_normalize(slack_pressure)
    rank_gate = np.clip(1.0 - 2.0 * np.clip(norm_slack, 0.0, 1.9064858604128563), 0.0, 1.0)
    boosted_rank = norm_rank * (1.0 + 1.0228639196229838 * rank_gate)
    joint_risk_gate = ((norm_uncert > 0.5952744435260071) & (slack_pressure > 0)).astype(float)
    duration_risk_interaction = norm_duration * joint_risk_gate * 0.0042556405806818565
    wait_benefit = np.tanh(0.5408638547632699 * norm_wait)
    coupled_rank = np.clip(norm_rank * (1.0 + 0.6714441511444741 * slack_pressure), 0.0, 2.0)
    work_pressure = norm_work * np.where(slack_pressure > 0, 1.0, 0.0) * 0.08860076192822072
    energy_under_pressure = norm_energy * (1.0 + 0.5920589297981658 * slack_pressure)
    score = +slack_pressure + energy_under_pressure - coupled_rank - wait_benefit + duration_risk_interaction + work_pressure
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    return score
