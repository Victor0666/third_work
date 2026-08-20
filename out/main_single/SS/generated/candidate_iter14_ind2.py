import numpy as np
RULE_METADATA = {'structure_hash': 'c0266351a7c99fcca4e951e1701b6714b40962c7284591f42e3b0f702250b85a', 'parameter_schema_hash': '6a07ce95908a9da419b304f020435585e5bb5b992d98ffaeade481f5df9a9ec2', 'best_parameter_hash': 'fb6b0526101a5fbd570a861c2517207c5765cec6081eba558c4dcce6147b9be9', 'best_parameters': {'epsilon': 0.009844639816190404, 'criticality_scale': 0.5246543726832322, 'energy_sensitivity': 0.1848421867787286, 'energy_uncertainty_interaction': 0.9981688502784364, 'remaining_work_weight': 0.48306086102315327, 'ddl_protection_gate_slope': 4.49550462272206, 'uncertainty_gate_threshold': 0.025340906263259067, 'wait_decay': 0.6291158053089491, 'slack_pressure_steepness': 4.9426312174337745, 'successor_release_weight': 0.8836462556719553}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '04fb614d0ce22e4d88168c605b7e4209a61512788b61e7472b8256183a22827f', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: merges Parent 2's numerical stability with Parent 1's structured slack-pressure gating;
       introduces novel sigmoid-based slack-pressure gate for critical-path coupling — smoother and more discriminative than linear ramp;
       retains Parent 2's strict DDL-protection gate to disable all risk-sensitive penalties when slack < 0;
       replaces raw duration penalty with robustified duration-uncertainty product under DDL pressure;
       adds explicit successor-release weight parameter for calibrated downstream impact;
       uses median-MAD normalization throughout for outlier resilience and sign preservation."""
    eps = 0.009844639816190404
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
    ddl_gate = np.clip(slack, 0.0, 1.0)
    ddl_gate = np.where(slack < 0.0, 0.0, ddl_gate)
    slack_pressure = 1.0 / (1.0 + np.exp(-4.9426312174337745 * -norm_slack))
    successor_release = 0.8836462556719553 * norm_rank * norm_work * slack_pressure
    coupled_rank = norm_rank * (1.0 + 0.5246543726832322 * slack_pressure)
    uncert_gate = np.where(norm_uncert > 0.025340906263259067, 1.0, 0.0)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    wait_benefit = np.clip(0.6291158053089491 * ready_wait_time, 0.0, 1.0)
    duration_uncert_penalty = norm_duration * norm_uncert * slack_pressure * ddl_gate
    raw_slack_penalty = np.maximum(-slack, 0.0)
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - np.clip(coupled_rank, -2.0, 2.0) - 0.1848421867787286 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(duration_uncert_penalty, -2.0, 2.0) + 0.9981688502784364 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 0.48306086102315327 * np.clip(norm_work, -2.0, 2.0) + np.clip(slack_pressure * 4.49550462272206, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
