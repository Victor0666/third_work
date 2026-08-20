import numpy as np
RULE_METADATA = {'structure_hash': '05c9b25baa48be5ec8435ec683d26b942d0f92bc1e83fe8dc2a17c280525653b', 'parameter_schema_hash': '4d5e934adbc19f85c54117682cfb79b5174e65de908df328af2109bf7aa43efe', 'best_parameter_hash': 'ff0f5e0f82296bcb0e44f0e80b0592ce88fae28c46cedf8a51f442b4ce3ecee0', 'best_parameters': {'epsilon': 0.0011725468141000376, 'slack_penalty_exponent': 1.0008954185489654, 'criticality_scale': 1.1451302647227397, 'energy_sensitivity': 0.10058483939325669, 'energy_uncertainty_interaction': 0.4389460103034467, 'ddl_protection_gate_slope': 5.892904757621971, 'remaining_work_weight': 0.5806845175565348, 'uncertainty_gate_threshold': 0.4103346554027387, 'wait_decay': 0.2777543713488322}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '87bd0fb87cf96005da93cab9309edcbaa1f3def6c9f847de08db2ef2d7a73aae', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule: replaces sigmoid DDL gates with bounded smooth pressure function;
       introduces successor-release interaction via min_exec_time × upward_rank × (slack <= 0);
       replaces median-MAD with robust MAD-only scaling (no centering) to preserve absolute urgency;
       removes redundant duration_robustness and wait_saturation_offset per diagnostics;
       enforces strict feasibility-first via hard binary ddl_gate for energy/uncertainty terms;
       adds host-load surrogate: norm_duration × norm_uncert gated by both slack >= 0 and uncertainty > threshold."""
    eps = 0.0011725468141000376
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def mad_normalize(x):
        x = np.copy(x)
        if N == 1:
            mad = eps
        else:
            mad = np.median(np.abs(x - np.median(x)))
        spread = mad if mad > eps else eps
        return x / spread
    norm_slack = mad_normalize(slack)
    norm_energy = mad_normalize(min_incremental_energy)
    norm_duration = mad_normalize(min_exec_time + min_comm_time)
    norm_rank = mad_normalize(upward_rank)
    norm_work = mad_normalize(remaining_work)
    norm_wait = mad_normalize(ready_wait_time)
    norm_uncert = mad_normalize(uncertainty)
    ddl_gate = (slack >= 0.0).astype(float)
    ddl_breach = (slack < 0.0).astype(float)
    critical_path_leverage = norm_rank * norm_work * ddl_breach
    successor_release = min_exec_time * norm_rank * ddl_breach
    norm_successor_release = mad_normalize(successor_release)
    ddl_pressure = 1.0 / (1.0 + np.abs(slack) + eps)
    load_surrogate_gate = (uncertainty > 0.4103346554027387).astype(float) * ddl_gate
    host_load_surrogate = norm_duration * norm_uncert * load_surrogate_gate
    wait_benefit = 1.0 - np.exp(-0.2777543713488322 * ready_wait_time)
    wait_benefit = np.clip(wait_benefit, 0.0, 1.0 - eps)
    energy_uncert_gate = load_surrogate_gate
    energy_uncert_penalty = norm_energy * norm_uncert * energy_uncert_gate
    score = +np.clip(norm_slack, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(norm_successor_release, -2.0, 2.0) - 0.10058483939325669 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.5806845175565348 * np.clip(norm_work, -2.0, 2.0) + 5.892904757621971 * np.clip(host_load_surrogate, -2.0, 2.0) + 0.4389460103034467 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 1.0008954185489654 * np.clip(ddl_pressure, -2.0, 2.0) + 1.1451302647227397 * np.clip(norm_rank * ddl_pressure, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
