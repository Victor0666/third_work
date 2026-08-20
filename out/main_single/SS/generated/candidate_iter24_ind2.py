import numpy as np
RULE_METADATA = {'structure_hash': '9e39428a14cd5b102f52a1fba504745f07046477bd23d61601f5cbe9cf6fc0f0', 'parameter_schema_hash': 'cff8f203b5001b6648fd8db8af0b8e814a10b4714c0b5d5150b2d5039d3084c7', 'best_parameter_hash': '3c0a6e08457431ae49dce9d9035bf11c0e57c4e8ad4c770dc68691af9044e90a', 'best_parameters': {'epsilon': 0.0007367938403742415, 'criticality_scale': 2.124665675587962, 'energy_sensitivity': 0.5180674235023531, 'energy_uncertainty_interaction': 0.21665815831939295, 'remaining_work_weight': 1.4085094666241624, 'uncertainty_gate_threshold': 0.401965372956265, 'wait_decay': 0.518441531145371, 'slack_penalty_linear_coeff': 0.5116579934528664, 'duration_robustness_factor': 0.48864915692710276, 'slack_energy_suppression_strength': 0.8063686396910829, 'ddl_protection_gate_slope': 4.819281304375302, 'host_load_sensitivity': 0.6741469978068543}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '9b951cd49ca3b68c3fe2c747ca77c0c04f9bc5384061e7c8d903421b73a71f45', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: reinstates HARD binary feasibility gating (not sigmoidal) per reflection;
       unifies all features under robust MAD normalization (including slack) for outlier resilience;
       promotes explicit successor-release coupling as additive term — now *unweighted* to avoid parameter bloat;
       removes redundant sigmoid-derived gates and preserves sharp deadline boundary behavior;
       introduces bounded linear wait_benefit modulation by (1 - ddl_pressure) *only* under feasibility."""
    eps = 0.0007367938403742415
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
    ddl_pressure = np.clip(-norm_slack, 0.0, 1.0)
    raw_slack_penalty = np.maximum(-slack, 0.0)
    norm_slack_penalty = robust_normalize(raw_slack_penalty)
    slack_penalty = 0.5116579934528664 * norm_slack_penalty
    successor_release = norm_rank * norm_work * ddl_pressure
    coupled_rank = norm_rank * (1.0 + 2.124665675587962 * ddl_pressure)
    uncert_gate = np.where(norm_uncert > 0.401965372956265, 1.0, 0.0)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * hard_feasibility_gate
    wait_benefit = np.clip(0.518441531145371 * ready_wait_time * (1.0 - ddl_pressure) * hard_feasibility_gate, 0.0, 2.0)
    duration_penalty = 0.48864915692710276 * norm_duration * ddl_pressure * hard_feasibility_gate
    slack_energy_suppress = np.where(slack < 0.0, 1.0 - 0.8063686396910829, 1.0)
    suppressed_norm_energy = norm_energy * slack_energy_suppress
    load_surrogate = norm_duration * norm_uncert
    load_gate = np.where((hard_feasibility_gate > 0.0) & (norm_uncert > 0.401965372956265), np.clip(load_surrogate, 0.0, 2.0), 0.0)
    host_load_penalty = 0.6741469978068543 * load_gate * suppressed_norm_energy
    ddl_protection_boost = np.clip(4.819281304375302 * ddl_pressure, -2.0, 2.0)
    score = +np.clip(slack_penalty, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - np.clip(coupled_rank, -2.0, 2.0) - 0.5180674235023531 * np.clip(suppressed_norm_energy, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(duration_penalty, -2.0, 2.0) + 0.21665815831939295 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 1.4085094666241624 * np.clip(norm_work, -2.0, 2.0) + ddl_protection_boost + host_load_penalty
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.reshape(-1)
