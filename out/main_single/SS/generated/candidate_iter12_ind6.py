import numpy as np
RULE_METADATA = {'structure_hash': '550cce08ae3bf235451eddca82a073958d8378a1526dc0034296e25f04d8a1be', 'parameter_schema_hash': '3e87229e6a30dfc471613e2e984e078da9d19b8537acfb92606c67eb422e7968', 'best_parameter_hash': 'cca03811a16bea4238fd1636d3332e5361d7005fef914d3e998f06215542e090', 'best_parameters': {'epsilon': 0.001256922156541642, 'slack_penalty_exponent': 2.348141614028637, 'criticality_scale': 2.033319430169641, 'energy_sensitivity': 0.14812297323986218, 'slack_threshold_gate_slope': 2.843644777007376, 'slack_threshold_offset': 0.892121135037582, 'wait_decay': 0.38063100389707005, 'uncertainty_gate_threshold': 0.6452751361573503, 'slack_pressure_gate_steepness': 3.0595387510568868, 'remaining_work_weight': 0.029194201866881198, 'energy_uncertainty_interaction': 0.22168022007562438, 'successor_release_coupling': 0.11998128491795682}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': 'e8927f3dffce7afaa5afd8d8c35785e930495d1697f1dfbda81e7a2cd2ab5393', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: replaces fixed zero-slack DDL gate with adaptive threshold gate;
       introduces conditional successor-release interaction for blocked critical paths;
       uses rank-based quantile normalization for robust ordinal preservation under skew;
       removes inactive parameters (duration_robustness, wait_saturation_offset) and reallocates budget."""
    eps = 0.001256922156541642
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def quantile_normalize(x):
        x = np.copy(x)
        if N == 1:
            return np.zeros_like(x)
        ranks = np.argsort(np.argsort(x))
        percentile = (ranks + 1.0) / (N + 1.0)
        return 2.0 * percentile - 1.0
    norm_slack = quantile_normalize(slack)
    norm_energy = quantile_normalize(min_incremental_energy)
    norm_duration = quantile_normalize(min_exec_time + min_comm_time)
    norm_rank = quantile_normalize(upward_rank)
    norm_work = quantile_normalize(remaining_work)
    norm_wait = quantile_normalize(ready_wait_time)
    norm_uncert = quantile_normalize(uncertainty)
    adaptive_slack = slack - 0.892121135037582
    ddl_gate = 1.0 / (1.0 + np.exp(-2.843644777007376 * adaptive_slack))
    breach_mask = (slack <= 0.0).astype(float)
    release_coupling = np.exp(-0.11998128491795682 * remaining_work / (np.abs(slack) + eps))
    successor_release_boost = norm_rank * release_coupling * breach_mask
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.348141614028637
    norm_slack_penalty = quantile_normalize(raw_slack_penalty)
    coupled_slack = np.clip(norm_slack, -1.0, 1.0)
    rank_slack_coupling = 1.0 + 2.033319430169641 * coupled_slack
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-3.0595387510568868 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * rank_slack_coupling * (1.0 + 2.033319430169641 * rank_gate)
    uncert_gate = 1.0 / (1.0 + np.exp(-2.843644777007376 * (norm_uncert - 0.6452751361573503)))
    duration_risk_score = norm_duration * uncert_gate * slack_pressure * ddl_gate
    wait_benefit = 1.0 - np.exp(-0.38063100389707005 * ready_wait_time)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    energy_slack_penalty = norm_energy * (1.0 + 2.348141614028637 * slack_pressure) * ddl_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(successor_release_boost, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 0.14812297323986218 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(duration_risk_score, -2.0, 2.0) + 0.22168022007562438 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 2.348141614028637 * np.clip(energy_slack_penalty, -2.0, 2.0) + 0.029194201866881198 * np.clip(norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
