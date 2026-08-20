import numpy as np
RULE_METADATA = {'structure_hash': '55b2b238f3cb1d50a43aa8232559b14781fd71187d2491ce919ea884b0c4ed5f', 'parameter_schema_hash': '6e40841b8843f97d8e9c64ec3415c8b4db90797a20b02a1e16a937ed76d05847', 'best_parameter_hash': '7b6cb09edeb19097b450c212b2f9e52d3c3acfbd047c7541b71d90e77211f3bb', 'best_parameters': {'epsilon': 3.585274575452322e-05, 'slack_penalty_exponent': 1.957086521417877, 'criticality_scale': 1.6324023116831667, 'energy_sensitivity': 1.1196438918751919, 'wait_activation_threshold_ratio': 0.4592051808340833, 'wait_benefit_steepness': 4.2049631107043215, 'uncertainty_gate_threshold': 0.5951586083288961, 'slack_pressure_gate_steepness': 1.421606471087489, 'remaining_work_weight': 0.5532500925823992, 'wait_saturation_offset': 4.6833045606123124e-07, 'energy_uncertainty_interaction': 0.9998621418903912, 'ddl_protection_gate_slope': 6.802551450156964}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '7dea2c02fbabcb8cc017730de68265c394f72299b34d98ff452efa9460c1d9c6', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: replaces global starvation relief with *conditional* anti-starvation gate — activated only when a task has waited longer than threshold_ratio × median_wait, preventing overcorrection;
       retains decoupled exec/comm signals and median-MAD normalization;
       introduces wait_benefit_steepness for sharper, more controllable saturation;
       removes redundant wait_decay and comm_priority_under_low_uncertainty (merged into existing logic via norm_comm + ddl_gate);
       all intermediate terms clipped to [-2,2] for bounded AST depth and stability."""
    eps = 3.585274575452322e-05
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def median_mad_normalize(x):
        x = np.copy(x)
        if N == 1:
            med = x[0]
            mad = eps
        else:
            med = np.median(x)
            mad = np.median(np.abs(x - med))
        spread = mad if mad > eps else eps
        return (x - med) / spread
    norm_slack = median_mad_normalize(slack)
    norm_energy = median_mad_normalize(min_incremental_energy)
    norm_exec = median_mad_normalize(min_exec_time)
    norm_comm = median_mad_normalize(min_comm_time)
    norm_rank = median_mad_normalize(upward_rank)
    norm_work = median_mad_normalize(remaining_work)
    norm_wait = median_mad_normalize(ready_wait_time)
    norm_uncert = median_mad_normalize(uncertainty)
    ddl_gate = 1.0 / (1.0 + np.exp(-6.802551450156964 * slack))
    raw_slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    slack_pressure_gate = 1.0 / (1.0 + np.exp(-1.421606471087489 * (raw_slack_pressure - 1.0)))
    critical_path_booster = norm_rank * norm_work * ddl_gate
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.957086521417877
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    rank_amplifier = 1.0 + 1.6324023116831667 * slack_pressure_gate
    rank_amplifier = np.clip(rank_amplifier, 1.0, 2.0)
    boosted_rank = norm_rank * rank_amplifier
    uncert_gate = 1.0 / (1.0 + np.exp(-6.802551450156964 * (norm_uncert - 0.5951586083288961)))
    wait_median = np.median(ready_wait_time) if N > 1 else ready_wait_time[0]
    wait_threshold = 0.4592051808340833 * (wait_median + 4.6833045606123124e-07)
    wait_activation = 1.0 / (1.0 + np.exp(-4.2049631107043215 * (ready_wait_time - wait_threshold)))
    wait_benefit = wait_activation * (1.0 - np.exp(-1.0 * (ready_wait_time - wait_threshold + 4.6833045606123124e-07)))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    energy_slack_penalty = norm_energy * (1.0 + 1.957086521417877 * raw_slack_pressure) * ddl_gate
    exec_penalty = norm_exec * slack_pressure_gate * ddl_gate
    comm_penalty = norm_comm * ddl_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_booster, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 1.1196438918751919 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.9998621418903912 * np.clip(energy_uncert_penalty, -2.0, 2.0) + np.clip(energy_slack_penalty, -2.0, 2.0) + 0.5532500925823992 * np.clip(norm_work, -2.0, 2.0) + np.clip(exec_penalty, -2.0, 2.0) + np.clip(comm_penalty, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
