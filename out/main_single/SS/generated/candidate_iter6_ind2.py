import numpy as np
RULE_METADATA = {'structure_hash': '4b8b39b61fdba125499a780acc576ae569759e780d55e8d2d343c8515f4af7bf', 'parameter_schema_hash': 'bd72bc016789321a87d9b384cdf2783c70ce9c0ee8a568f260ec280e1cea9177', 'best_parameter_hash': 'c0ac8c9d6293f0006d35d2f823410caf396df1da27d0ee77032414eb82417802', 'best_parameters': {'epsilon': 0.010476046029620396, 'slack_penalty_exponent': 2.896174734478568, 'criticality_scale': 0.9884948527773001, 'energy_sensitivity': 0.28181279803355863, 'duration_robustness': 1.8210085958862585, 'wait_decay': 0.6537603215269006, 'uncertainty_gate_threshold': 0.6935984409691593, 'ddl_protection_steepness': 7.90738765762093, 'remaining_work_weight': 1.2715172263398515, 'wait_saturation_offset': 1.000162379573146e-09, 'energy_uncertainty_interaction': 0.45282711895896754, 'ddl_priority_bias': 6.530153389218515}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '06650f6662e6156569ae4240a920b1035ed02e87b30c3c106fe67ce3b5f86e19', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: introduces a hard-deadline protection gate that *dominates* the score when slack < 0,
       using a steep sigmoid to trigger additive priority bias — ensuring strict DDL satisfaction over energy minimization.
       Replaces rank-slack coupling with unified DDL gate; uses raw slack for fidelity *and* bounded normalized pressure
       for robust gating. All operations are finite, deterministic, and shape-compliant."""
    eps = 0.010476046029620396
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
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.896174734478568
    slack_pressure_raw = np.clip(-slack, 0.0, None)
    slack_pressure_norm = np.zeros_like(slack_pressure_raw)
    if N > 1:
        denom = np.max(slack_pressure_raw) - np.min(slack_pressure_raw) + eps
        slack_pressure_norm = np.clip((slack_pressure_raw - np.min(slack_pressure_raw)) / denom, 0.0, 2.0)
    else:
        slack_pressure_norm[0] = 0.0 if slack[0] >= 0 else 2.0
    ddl_gate = 1.0 / (1.0 + np.exp(-7.90738765762093 * (slack_pressure_norm - 1.0)))
    boosted_rank = norm_rank * (1.0 + 0.9884948527773001 * ddl_gate)
    uncert_gate = np.where(norm_uncert > 0.6935984409691593, 1.0, 0.0)
    duration_risk_score = norm_duration * uncert_gate * slack_pressure_norm
    wait_benefit = 1.0 - np.exp(-0.6537603215269006 * (norm_wait + 1.000162379573146e-09))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate
    base_score = +raw_slack_penalty - boosted_rank - 0.28181279803355863 * norm_energy - norm_duration - wait_benefit + 1.8210085958862585 * duration_risk_score + 0.45282711895896754 * energy_uncert_penalty + 1.2715172263398515 * norm_work
    ddl_override = 6.530153389218515 * ddl_gate
    score = base_score + ddl_override
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
