import numpy as np
RULE_METADATA = {'structure_hash': 'b5bd4c5b267647457a545f214cb5323471046d674ddee46f0ec7773634f61416', 'parameter_schema_hash': '014247cafd2931086feb479f0011584c6f4336afe1f0dccb05c2a9a66300a56b', 'best_parameter_hash': 'a6e0cc9e9006ceae06c70d9babb4d68a8a25672913486a3d5bf035275e6f4496', 'best_parameters': {'epsilon': 0.0009435199966982932, 'slack_penalty_exponent': 3.9012547981125145, 'criticality_scale': 2.4571770372702204, 'energy_sensitivity': 0.10460837266010498, 'comm_energy_coupling_weight': 0.7083473573880972, 'wait_decay': 0.3805445205630562, 'uncertainty_gate_threshold': 0.6063408495640632, 'slack_pressure_gate_steepness': 3.7256144772016713, 'remaining_work_weight': 0.3069638621834147, 'successor_release_steepness': 4.952766645286939, 'energy_uncertainty_interaction': 0.791259677385452, 'ddl_protection_gate_slope': 3.854248925156396}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '66d3706c1ca65554920216eca79132e24914c5890ed0ddb4fb40d60598df747c', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: replaces deprecated duration_robustness with active comm-energy coupling;
       introduces successor-release gate to prioritize unblocking of critical downstream work;
       removes inactive wait_saturation_offset and uses robust logistic starvation relief;
       retains median-MAD normalization and strict DDL-first semantics via dual-gated terms."""
    eps = 0.0009435199966982932
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
    norm_comm_energy = median_mad_normalize(min_comm_time * min_incremental_energy)
    ddl_gate = 1.0 / (1.0 + np.exp(-3.854248925156396 * slack))
    ddl_breach = (slack <= 0.0).astype(float)
    critical_path_leverage = norm_rank * norm_work * ddl_breach
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 3.9012547981125145
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-3.7256144772016713 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * (1.0 + 2.4571770372702204 * rank_gate)
    release_horizon = remaining_work / (upward_rank + eps)
    successor_release_gate = 1.0 / (1.0 + np.exp(-4.952766645286939 * (slack - release_horizon)))
    wait_benefit = 1.0 - np.exp(-0.3805445205630562 * ready_wait_time)
    uncert_gate = 1.0 / (1.0 + np.exp(-3.854248925156396 * (norm_uncert - 0.6063408495640632)))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    comm_energy_penalty = norm_comm_energy * ddl_gate * slack_pressure
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 0.10460837266010498 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.7083473573880972 * np.clip(comm_energy_penalty, -2.0, 2.0) + 0.791259677385452 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 0.3069638621834147 * np.clip(norm_work, -2.0, 2.0) - np.clip(successor_release_gate, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
