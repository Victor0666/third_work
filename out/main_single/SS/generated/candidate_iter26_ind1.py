import numpy as np
RULE_METADATA = {'structure_hash': '44997252ea4918316a3e938f2140fc302795b0395bd0eb106468645ca99b7436', 'parameter_schema_hash': '73ea271f8fb43f3fb404b50870bd417f229f25644763bcd57b03eb29ebdf42dc', 'best_parameter_hash': 'c5580b6f081f12487db57680a74c7f68010ce20c7cbca46f22399443a9a985c4', 'best_parameters': {'epsilon': 0.0002650229048545154, 'energy_sensitivity': 0.8035464586755646, 'energy_uncertainty_interaction': 0.7041521158185532, 'remaining_work_weight': 0.9941537876668214, 'uncertainty_gate_threshold': 0.37136420037794926, 'wait_decay': 0.9448296401923109, 'slack_penalty_linear_coeff': 2.2044563535412283, 'duration_robustness_factor': 0.5300034128849507, 'slack_energy_suppression_strength': 0.993032445862984, 'host_load_sensitivity': 0.5986530742285214, 'urgency_curvature': 0.7174551415406162}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '992504c6d42d781e37621492137a990ae150329236b5f81fb76bc792547d3900', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: merges robust median-MAD normalization with convex slack urgency;
       uses multiplicative critical-path coupling (rank × work × pressure);
       retains smooth ddl_pressure = 1/(1+|slack|+eps), congestion-aware host-load surrogate,
       and tunable energy suppression; enforces strict finite-range clipping and shape (N,) output."""
    eps = 0.0002650229048545154
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
    raw_slack_penalty = np.maximum(-slack, 0.0) + eps
    convex_slack_penalty = np.power(raw_slack_penalty, 0.7174551415406162)
    norm_slack_penalty = robust_normalize(convex_slack_penalty)
    slack_penalty = 2.2044563535412283 * norm_slack_penalty
    critical_coupling = norm_rank * norm_work * ddl_pressure
    uncert_gate = np.where(norm_uncert > 0.37136420037794926, 1.0, 0.0)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * hard_feasibility_gate
    wait_benefit = np.clip(0.9448296401923109 * ready_wait_time, 0.0, 2.0)
    duration_penalty = 0.5300034128849507 * norm_duration * ddl_pressure * hard_feasibility_gate
    slack_energy_suppress = np.where(slack < 0.0, 1.0 - 0.993032445862984, 1.0)
    suppressed_norm_energy = norm_energy * slack_energy_suppress
    load_surrogate = norm_duration * norm_uncert
    load_gate = np.where((hard_feasibility_gate > 0.0) & (norm_uncert > 0.37136420037794926), np.clip(load_surrogate, 0.0, 2.0), 0.0)
    host_load_penalty = 0.5986530742285214 * load_gate * suppressed_norm_energy
    score = +np.clip(slack_penalty, -2.0, 2.0) - np.clip(critical_coupling, -2.0, 2.0) - 0.8035464586755646 * np.clip(suppressed_norm_energy, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(duration_penalty, -2.0, 2.0) + 0.7041521158185532 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 0.9941537876668214 * np.clip(norm_work, -2.0, 2.0) + host_load_penalty
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.reshape(-1)
