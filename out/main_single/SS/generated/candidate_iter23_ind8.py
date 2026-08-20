import numpy as np
RULE_METADATA = {'structure_hash': 'b4e703f1b7292bb9d5e65a24d1fcf74e156c0a989038f91ca2320f71d1d15d4b', 'parameter_schema_hash': '59da279f77c3b8b6c9427a44784c788331a62dd26ebb2b07f6cbf24ba840dc9a', 'best_parameter_hash': 'a4330ca3ebd2ad771670696c0941d31c2f1e13edaa6eee2ccfaaeaf5c7939eb1', 'best_parameters': {'epsilon': 0.0012728720220763876, 'slack_penalty_exponent': 1.0593151294305214, 'criticality_scale': 0.8193109436024343, 'energy_sensitivity': 0.49599726936146993, 'uncertainty_gate_threshold': 0.4956296749993053, 'slack_pressure_gate_steepness': 2.618624102353422, 'remaining_work_weight': 0.8079977736899158, 'wait_decay': 0.957358672572623, 'ddl_protection_gate_slope': 2.0422516641434685, 'energy_uncertainty_interaction': 0.24498505855180622, 'congestion_coupling_strength': 0.7149534259844921, 'starvation_relief_steepness': 0.5062977865387044}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': 'fe25d038f12efa4673759621fdad7e838110447748e708ca6086a28a61512a83', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: merges Parent 2's robust hinge coupling and joint gate with Parent 1's congestion-aware load signal;
       replaces softplus starvation relief with *steepened softplus* to sharpen low-wait sensitivity while preserving saturation;
       introduces *congestion_coupling_strength* to explicitly weight duration-uncertainty co-penalty only under joint feasibility;
       retains median-MAD normalization for outlier resistance and sign stability;
       removes redundant rank_slack_coupling amplification (redundant with slack_pressure_gate) to reduce AST depth;
       enforces strict clipping at [-2,2] on all composite terms to bound gradient magnitude and prevent dominance collapse."""
    eps = 0.0012728720220763876
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
    joint_gate = 1.0 / (1.0 + np.exp(-2.0422516641434685 * slack)) * 1.0 / (1.0 + np.exp(2.0422516641434685 * (norm_uncert - 0.4956296749993053)))
    ddl_breach = (slack <= 0.0).astype(float)
    critical_path_leverage = norm_rank * norm_work * ddl_breach
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.0593151294305214
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-2.618624102353422 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * (1.0 + 0.8193109436024343 * rank_gate)
    starvation_signal = 0.957358672572623 * (ready_wait_time + eps)
    wait_benefit = np.log1p(np.exp(0.5062977865387044 * starvation_signal))
    congestion_coupling = 0.7149534259844921 * norm_duration * norm_uncert * joint_gate
    energy_uncert_penalty = norm_energy * norm_uncert * joint_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 0.49599726936146993 * np.clip(norm_energy * joint_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(congestion_coupling, -2.0, 2.0) + 0.24498505855180622 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 0.8079977736899158 * np.clip(norm_work * joint_gate, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
