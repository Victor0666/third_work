import numpy as np
RULE_METADATA = {'structure_hash': '701ef6c5e1e4e4e4401d62653f7d5ea7c5ec589205b8e05304365d80b1c458e3', 'parameter_schema_hash': 'e78d98c9eb0358a7be09978b5171e0d613ff7b5f422cfcbd20a1724457b2ef5e', 'best_parameter_hash': '37999a3fbf7800a8da1dd02e85edf7a73400dbdd4397f4a4885253bb63dbc85c', 'best_parameters': {'epsilon': 0.016251102168383776, 'criticality_scale': 1.4123598118989062, 'energy_sensitivity': 0.19234406735086237, 'energy_uncertainty_interaction': 0.8401948840818059, 'remaining_work_weight': 0.2666893647654507, 'uncertainty_gate_threshold': 0.4970654130583055, 'wait_decay': 0.07169515704466178, 'slack_penalty_linear_coeff': 2.9530244658677973, 'duration_robustness_factor': 0.15295082508985988, 'slack_energy_suppression_strength': 0.7209296745651153, 'host_load_sensitivity': 0.6479595606061713, 'starvation_feasibility_coupling': 0.8866794462216384}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '712082e051c3ddcf7725b653298ef61e8b3eab5d16cca007d4fa62de8bad8dfc', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule: merges best of both parents — preserves Parent 2's stable inverse ddl_pressure,
       adopts Parent 1's feasibility-gated & pressure-modulated starvation relief, removes redundant slope parameter,
       introduces explicit starvation_feasibility_coupling for adaptive anti-starvation under tight deadlines,
       unifies all robust normalizations, enforces hard feasibility gating on energy/uncertainty terms,
       and maintains strict [-2,2] clipping per term for bounded AST depth and stability."""
    eps = 0.016251102168383776
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
    raw_slack_penalty = np.maximum(-slack, 0.0)
    norm_slack_penalty = robust_normalize(raw_slack_penalty)
    slack_penalty = 2.9530244658677973 * norm_slack_penalty
    successor_release = norm_rank * norm_work * ddl_pressure
    coupled_rank = norm_rank * (1.0 + 1.4123598118989062 * ddl_pressure)
    uncert_gate = np.where(norm_uncert > 0.4970654130583055, 1.0, 0.0)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * hard_feasibility_gate
    wait_benefit = np.clip(0.07169515704466178 * ready_wait_time * 0.8866794462216384 * hard_feasibility_gate * (1.0 - ddl_pressure), 0.0, 2.0)
    duration_penalty = 0.15295082508985988 * norm_duration * ddl_pressure * hard_feasibility_gate
    slack_energy_suppress = np.where(slack < 0.0, 1.0 - 0.7209296745651153, 1.0)
    suppressed_norm_energy = norm_energy * slack_energy_suppress
    load_surrogate = norm_duration * norm_uncert
    load_gate = np.where((hard_feasibility_gate > 0.0) & (norm_uncert > 0.4970654130583055), np.clip(load_surrogate, 0.0, 2.0), 0.0)
    host_load_penalty = 0.6479595606061713 * load_gate * suppressed_norm_energy
    score = +np.clip(slack_penalty, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - np.clip(coupled_rank, -2.0, 2.0) - 0.19234406735086237 * np.clip(suppressed_norm_energy, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(duration_penalty, -2.0, 2.0) + 0.8401948840818059 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 0.2666893647654507 * np.clip(norm_work, -2.0, 2.0) + host_load_penalty
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.reshape(-1)
