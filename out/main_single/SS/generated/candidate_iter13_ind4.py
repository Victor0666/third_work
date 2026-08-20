import numpy as np
RULE_METADATA = {'structure_hash': '4da589a641a2f94908980c92d77806fd91294ea03bf45008332112d55fb8e5b3', 'parameter_schema_hash': '20bef682e2def978ccf10644e2e8fd52b5012787b2ff92ab6f1573ff4ac04a2d', 'best_parameter_hash': '09a16e6c19634c504a404269186b628dac49155242a589faf4eaa23f9d953718', 'best_parameters': {'epsilon': 0.00014071584717196696, 'criticality_scale': 0.5576225274428079, 'energy_sensitivity': 1.511779525826941, 'energy_uncertainty_interaction': 0.5967942827302208, 'remaining_work_weight': 0.02355996812511357, 'slack_pressure_gate_steepness': 1.20223889489514, 'uncertainty_gate_threshold': 0.25492832188020054, 'wait_decay': 0.6904465800486197, 'linear_interpolation_bias': 0.8406272546744726, 'wait_saturation_cap': 0.9886492515625837}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': 'b76195b68595f167c076b6cb5c3826d6c33076135c5c20cbf51432603cc539eb', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule: replaces sigmoid gates with bounded linear interpolation for monotonic stability;
       removes redundant duration_robustness and wait_saturation_offset (evidence shows near-zero variance);
       introduces successor-release coupling via norm_work * norm_rank interaction, gated by slack pressure;
       uses clipped linear slack mapping instead of power-law for better numerical behavior under tight DDL;
       applies median-MAD normalization per feature with N-aware fallback; enforces deterministic finite output."""
    eps = 0.00014071584717196696
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
    norm_rank = median_mad_normalize(upward_rank)
    norm_work = median_mad_normalize(remaining_work)
    norm_wait = median_mad_normalize(ready_wait_time)
    norm_uncert = median_mad_normalize(uncertainty)
    ddl_feasible = (slack >= 0.0).astype(float)
    slack_pressure = np.clip(norm_slack, -1.0, 1.0)
    pressure_gate = 0.8406272546744726 * (1.0 + slack_pressure)
    successor_coupling = norm_work * norm_rank * pressure_gate
    critical_boost = norm_rank * (1.0 + 0.5576225274428079 * pressure_gate)
    wait_benefit = 1.0 - np.exp(-0.6904465800486197 * ready_wait_time)
    wait_benefit = np.clip(wait_benefit, 0.0, 0.9886492515625837)
    uncert_gate = (norm_uncert > 0.25492832188020054).astype(float)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_feasible
    energy_slack_sensitivity = 1.511779525826941 * (1.0 + 1.20223889489514 * (1.0 - pressure_gate))
    score = +np.clip(-norm_slack, -2.0, 2.0) - np.clip(critical_boost, -2.0, 2.0) - np.clip(successor_coupling, -2.0, 2.0) - energy_slack_sensitivity * np.clip(norm_energy * ddl_feasible, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.5967942827302208 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 0.02355996812511357 * np.clip(norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
