import numpy as np
RULE_METADATA = {'structure_hash': 'c48851ab6cd354f91f7548270f04e6c34ea2ed380af39c0f3838d2578a3b43be', 'parameter_schema_hash': '3af448ababcea14758406d7b001ab094e04c11c52bc9a5d387a1f9339093ff57', 'best_parameter_hash': 'd7a37afca32763a26a2b85660df1c104005d0b04a33cccb6518118bfccb8bf4e', 'best_parameters': {'epsilon': 1.1467229206703386e-06, 'criticality_scale': 0.5957840926538829, 'energy_sensitivity': 1.0988308681144363, 'energy_uncertainty_interaction': 0.7250293386329227, 'remaining_work_weight': 1.1015586381148403, 'ddl_protection_gate_slope': 2.0337304589939653, 'uncertainty_gate_threshold': 0.2515619333426803, 'wait_decay': 0.49528025734920783, 'slack_pressure_steepness': 7.814070446325218, 'safe_slack_energy_disable_threshold': 0.34584613536241593}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': 'c3653f8866f35e5da63a3dcd7ce3afd8fe10316e20d8b2baf91cd2f5dd176db9', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule: hybridizes Parent 2's stability with Parent 1's safe-slack gating.
       Novel structural change: dual-gate energy suppression — (i) hard DDL-protection gate (slack < 0 → disable), 
       and (ii) soft safe-slack gate (norm_slack > threshold → suppress), enabling robust feasibility-first behavior
       across all slack regimes. Introduces sigmoid-coupled successor-release interaction for sharper critical-path activation.
       Replaces linear ddl_gate with clipped sigmoid on normalized slack to improve gradient continuity while avoiding overflow."""
    eps = 1.1467229206703386e-06
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
    hard_ddl_gate = np.where(slack < 0.0, 0.0, 1.0)
    safe_slack_mask = norm_slack > 0.34584613536241593
    soft_energy_suppress = np.where(safe_slack_mask, 0.0, 1.0)
    energy_enabled = hard_ddl_gate * soft_energy_suppress
    slack_pressure_norm = np.clip(-norm_slack, 0.0, 2.0)
    slack_pressure = 1.0 / (1.0 + np.exp(-7.814070446325218 * (slack_pressure_norm - 1.0)))
    successor_release = norm_rank * norm_work * slack_pressure
    raw_slack_penalty = np.maximum(-slack, 0.0)
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    coupled_rank = norm_rank * (1.0 + 0.5957840926538829 * slack_pressure)
    uncert_gate = np.where(norm_uncert > 0.2515619333426803, 1.0, 0.0)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * energy_enabled
    wait_benefit = np.clip(0.49528025734920783 * ready_wait_time, 0.0, 1.0)
    exec_penalty = norm_duration * slack_pressure * hard_ddl_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - np.clip(coupled_rank, -2.0, 2.0) - 1.0988308681144363 * np.clip(norm_energy * energy_enabled, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(exec_penalty, -2.0, 2.0) + 0.7250293386329227 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 1.1015586381148403 * np.clip(norm_work, -2.0, 2.0) + np.clip(slack_pressure * 2.0337304589939653, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
