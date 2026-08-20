import numpy as np
RULE_METADATA = {'structure_hash': '44e8ee5533c9f04b4b22d5a56240efcbf77c42f8c2ffec711617df45bb47025b', 'parameter_schema_hash': '7461f2949caeb92f8b171a0b11e40af3e1d8f0e9bec31458188e925126247bcd', 'best_parameter_hash': '7c94f4530489c8ead542fab9d76cbbd4dcbd7a55c5be2e224b6c3d6ca3d304ba', 'best_parameters': {'epsilon': 2.8017192444868692e-05, 'criticality_scale': 0.5052598355171528, 'energy_sensitivity': 0.3385217497183796, 'energy_uncertainty_interaction': 0.7739906368535561, 'remaining_work_weight': 0.43313596411772926, 'ddl_protection_gate_slope': 7.465484598562245, 'uncertainty_gate_threshold': 0.5278851571772669, 'wait_decay': 0.305359747299619, 'slack_penalty_linear_coeff': 1.2477046615893523, 'slack_energy_suppression_strength': 0.7673945596127526, 'load_aware_suppression_strength': 0.4641027836404151}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '2aff7ca53c985d141574283f6a789f668ac53bd49b85a68a156d0e84b862bb9a', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: restores `ddl_protection_gate_slope` and introduces `load_aware_suppression_strength`
       to dynamically suppress energy preference on high-duration tasks — directly targeting 'avoidable_marginal_energy_or_load_cost'.
       Removes unstable `slack_alignment_exponent` per reflection, reverting to robust linear slack penalty + gated critical-path amplification.
       Introduces duration-load gate: `np.clip((min_exec_time + min_comm_time) / (median_duration + eps), 0.0, 2.0)` to identify congested candidates.
       All energy-related contributions now modulated by both hard feasibility AND duration-load gate — preventing over-aggressive low-energy scheduling on already-loaded VMs.
       Maintains hard slack-driven energy suppression and unified robust normalization."""
    eps = 2.8017192444868692e-05
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
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    duration = min_exec_time + min_comm_time
    if N == 1:
        median_duration = duration[0]
    else:
        median_duration = np.median(duration)
    duration_load_signal = np.clip(duration / (median_duration + eps), 0.0, 2.0)
    hard_feasibility_gate = np.where(slack < 0.0, 0.0, 1.0)
    ddl_pressure = np.clip(-norm_slack, 0.0, 1.0)
    raw_slack_penalty = np.maximum(-slack, 0.0)
    norm_slack_penalty = robust_normalize(raw_slack_penalty)
    slack_penalty = 1.2477046615893523 * norm_slack_penalty
    successor_release = norm_rank * norm_work * ddl_pressure
    ddl_gate = 1.0 / (1.0 + np.exp(-7.465484598562245 * ddl_pressure))
    coupled_rank = norm_rank * (1.0 + 0.5052598355171528 * ddl_gate)
    uncert_gate = np.where(norm_uncert > 0.5278851571772669, 1.0, 0.0)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * hard_feasibility_gate
    wait_benefit = np.clip(0.305359747299619 * ready_wait_time, 0.0, 2.0)
    load_suppress = 1.0 - 0.4641027836404151 * (duration_load_signal - 1.0)
    load_suppress = np.clip(load_suppress, 0.0, 1.0)
    slack_energy_suppress = np.where(slack < 0.0, 1.0 - 0.7673945596127526, 1.0)
    suppressed_norm_energy = norm_energy * slack_energy_suppress * load_suppress * hard_feasibility_gate
    score = +np.clip(slack_penalty, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - np.clip(coupled_rank, -2.0, 2.0) - 0.3385217497183796 * np.clip(suppressed_norm_energy, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.7739906368535561 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 0.43313596411772926 * np.clip(norm_work, -2.0, 2.0)
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.reshape(-1)
