import numpy as np
RULE_METADATA = {'structure_hash': '8e2caea4d4a2814d1e4eab6e5dc96beee5f39069d2b2f26ac79986834e9b1dc0', 'parameter_schema_hash': '59de27546f5aa05d99fb5b6a6cdbed7454af38e6b118609ad5b9aef9761d301b', 'best_parameter_hash': '9b905216c983df34eb8b74207a43b4c7a7ef6949faa5e7688bbf01e9702fd8ba', 'best_parameters': {'epsilon': 0.08048164167197917, 'slack_penalty_exponent': 2.8477577432935157, 'criticality_scale': 1.5582520424711606, 'energy_sensitivity': 0.606220545677115, 'energy_uncertainty_interaction': 0.9902364284369594, 'remaining_work_weight': 1.8654698492719541, 'slack_pressure_gate_steepness': 1.3282000077873277, 'uncertainty_gate_threshold': 0.14372281848787388, 'wait_decay': 0.004809997120426398, 'congestion_wait_offset': 0.4460537319217166}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '576903631666af8f6c0d4b2b87dadd589b2a8522701eaf6b03132478e34ab49c', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule: replaces sigmoid DDL gates with bounded smooth pressure function;
       introduces successor-release interaction (min_exec_time * upward_rank * (slack <= 0));
       removes redundant duration_robustness and wait_saturation_offset (validated inactive);
       uses sign-preserving MAD-normalized slack for sharper urgency near zero;
       applies joint congestion gating only when both ready_wait_time and uncertainty are high;
       enforces strict feasibility-first energy penalization via hard binary gate (slack >= 0);
       adds host-load surrogate (norm_duration * norm_uncert) gated by feasibility and uncertainty threshold."""
    eps = 0.08048164167197917
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
    feasible_gate = (slack >= 0.0).astype(float)
    ddl_pressure = 1.0 / (1.0 + np.abs(slack) + eps)
    critical_breach = (slack <= 0.0).astype(float)
    critical_path_impact = norm_rank * norm_work * critical_breach
    clipped_rank = np.clip(norm_rank, 0.0, 2.0)
    successor_release = min_exec_time * clipped_rank * critical_breach
    wait_median = np.median(ready_wait_time)
    wait_std = np.std(ready_wait_time)
    wait_high = (ready_wait_time > wait_median + 0.4460537319217166 * wait_std).astype(float)
    uncert_high = (uncertainty > 0.14372281848787388).astype(float)
    congestion_gate = wait_high * uncert_high
    host_load_surrogate = norm_duration * norm_uncert * feasible_gate * uncert_high
    wait_benefit = 1.0 - np.exp(-0.004809997120426398 * ready_wait_time)
    energy_penalty = norm_energy * feasible_gate * (1.0 + 2.8477577432935157 * ddl_pressure)
    energy_uncert_coupling = norm_energy * norm_uncert * feasible_gate * uncert_high
    score = +np.clip(norm_slack, -2.0, 2.0) + np.clip(ddl_pressure, 0.0, 2.0) - np.clip(critical_path_impact, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - 0.606220545677115 * np.clip(energy_penalty, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.9902364284369594 * np.clip(energy_uncert_coupling, -2.0, 2.0) + 1.8654698492719541 * np.clip(norm_work, -2.0, 2.0) + 1.3282000077873277 * np.clip(host_load_surrogate, -2.0, 2.0) + 1.5582520424711606 * np.clip(norm_rank * ddl_pressure, -2.0, 2.0) + congestion_gate * np.clip(norm_wait + norm_uncert, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
