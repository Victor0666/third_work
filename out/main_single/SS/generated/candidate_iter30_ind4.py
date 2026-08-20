import numpy as np
RULE_METADATA = {'structure_hash': 'fc6e04ea88fd946494649afa29d2f4e38695de9e6bfdd730869d21ba220782d9', 'parameter_schema_hash': '847ca7ae5ef77b06804e70f7d9c3c6b25dd0af3e97d68b49cddd859aace4c1df', 'best_parameter_hash': '05440567f5df1b87128e7d2d317e579082151478bd7ead4a9ce579b4782f7721', 'best_parameters': {'epsilon': 0.002309525261207518, 'criticality_scale': 1.1057237791629824, 'energy_sensitivity': 0.10309972478499295, 'energy_uncertainty_interaction': 0.5853810014083347, 'remaining_work_weight': 1.2563672365314216, 'uncertainty_gate_threshold': 0.9002078480903255, 'wait_decay': 0.6094885613672483, 'ddl_pressure_gate_threshold': 0.9340991474775533, 'host_load_ramp_slope': 1.6700382340693034, 'host_load_ramp_offset': 0.23790192904942536}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '08da767421d40862c3279f9a3ab150cf7cc89946f32f194aabfa55e198deae61', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: replaces unstable sigmoid host-load gating with bounded linear ramp (robust, monotonic);
       restores signed MAD-normalized slack as primary urgency signal — avoids noise amplification from exponentiation;
       reinstates unconditional starvation relief (no slack sign gating) to prevent near-feasible starvation;
       retains robust ddl_pressure, successor-release, and critical-path urgency from prior versions;
       all feature interactions clipped to [-2,2] to enforce bounded AST depth and deterministic stability."""
    eps = 0.002309525261207518
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
    norm_duration = median_mad_normalize(min_exec_time + min_comm_time)
    norm_rank = median_mad_normalize(upward_rank)
    norm_work = median_mad_normalize(remaining_work)
    norm_wait = median_mad_normalize(ready_wait_time)
    norm_uncert = median_mad_normalize(uncertainty)
    ddl_pressure = 1.0 / (1.0 + np.abs(slack) + eps)
    ddl_breach = (slack <= 0.0).astype(float)
    critical_path_urgency = norm_rank * norm_work * ddl_breach
    successor_release = min_exec_time * norm_rank * ddl_breach
    norm_slack_penalty = np.clip(norm_slack, -2.0, 2.0)
    load_activation = norm_duration * norm_uncert
    host_load_ramp = np.clip(1.6700382340693034 * (load_activation - 0.23790192904942536), 0.0, 1.0)
    host_load_surrogate = load_activation * host_load_ramp * ddl_pressure
    energy_uncert_gate = (ddl_pressure > 0.9340991474775533).astype(float) * (norm_uncert >= 0.9002078480903255).astype(float)
    energy_uncert_penalty = norm_energy * norm_uncert * energy_uncert_gate
    wait_benefit = np.exp(-0.6094885613672483 * ready_wait_time) * (1.0 - ddl_pressure)
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_urgency, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - 0.10309972478499295 * np.clip(norm_energy * ddl_pressure, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(host_load_surrogate, -2.0, 2.0) + 0.5853810014083347 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 1.2563672365314216 * np.clip(norm_work, -2.0, 2.0) + 1.1057237791629824 * np.clip(norm_rank * ddl_pressure, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
