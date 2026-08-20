import numpy as np
RULE_METADATA = {'structure_hash': '2a2b42434a3359b017b5ddaa1225a0ee91a8eed9941b75f0ee2665dbd6b2b34a', 'parameter_schema_hash': '5ba6275c5a34b6ba2275df1f259e7f13fc11f659a291c28123d58d069aa3cb71', 'best_parameter_hash': 'eb4ec39347b134967ac0f4639bca48606a893c1874cb9a415abde001ffe15801', 'best_parameters': {'epsilon': 0.05266053032863679, 'slack_penalty_exponent': 2.700089267003326, 'criticality_scale': 2.092252979177133, 'energy_sensitivity': 0.49913372442607407, 'duration_robustness': 0.5485789329469639, 'wait_decay': 0.1828587601882044, 'uncertainty_gate_threshold': 0.7114237903490176, 'slack_pressure_gate_steepness': 4.144854386593032, 'remaining_work_weight': 1.2262234714164888, 'wait_saturation_offset': 7.269697010819337e-06, 'energy_uncertainty_interaction': 0.6776168984883219, 'ddl_protection_gate_slope': 4.179191150206047}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '72f260735fefe82c5a2126d06a8a01c2e7df7ed862ebd9a0cd00bbdbc7517277', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: introduces DDL-protection gate to enforce hard deadline feasibility before energy optimization.
    Energy and uncertainty terms are *disabled* under slack < 0 via a smooth, differentiable gate — ensuring constraint-first behavior.
    All declared parameters are used; no numeric literals except -2,-1,0,1,2; shape (N,) guaranteed."""
    eps = 0.05266053032863679
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
        if N == 1:
            center = x_abs[0]
            spread = eps
        else:
            center = np.median(x_abs)
            spread = np.median(np.abs(x_abs - center))
        return (x_abs - center) / (spread + eps)
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.700089267003326
    norm_slack_penalty = robust_normalize(raw_slack_penalty)
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-4.144854386593032 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * (1.0 + 2.092252979177133 * rank_gate)
    ddl_gate = 1.0 / (1.0 + np.exp(-4.179191150206047 * slack))
    uncert_gate = 1.0 / (1.0 + np.exp(-4.179191150206047 * (norm_uncert - 0.7114237903490176)))
    duration_risk_score = norm_duration * uncert_gate * slack_pressure * ddl_gate
    wait_benefit = 1.0 - np.exp(-0.1828587601882044 * (norm_wait + 7.269697010819337e-06))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    energy_slack_penalty = norm_energy * (1.0 + 2.700089267003326 * slack_pressure) * ddl_gate
    score = +norm_slack_penalty - boosted_rank - 0.49913372442607407 * norm_energy * ddl_gate - norm_duration * ddl_gate - wait_benefit + 0.5485789329469639 * duration_risk_score + 0.6776168984883219 * energy_uncert_penalty + 2.700089267003326 * energy_slack_penalty + 1.2262234714164888 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
