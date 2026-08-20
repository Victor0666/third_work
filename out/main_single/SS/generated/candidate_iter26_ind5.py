import numpy as np
RULE_METADATA = {'structure_hash': 'cef3d9434b4b8b409287202cd6ad98a60400c3d92072d9e19c4de47fc8b6dda3', 'parameter_schema_hash': '15a06554e8c20c46afd39337f6fcfc2f92f06d860a64ebac0e4173bb2dab9358', 'best_parameter_hash': 'fb66713c80b5d25b5d5588a0bc73ecf549b60c2e03e2e1fb83fb0d4be75913b8', 'best_parameters': {'epsilon': 0.004927901068561355, 'slack_penalty_exponent': 2.726339499843255, 'criticality_scale': 1.8467165687691889, 'energy_sensitivity': 1.8375025150459485, 'duration_robustness': 0.9432786612092171, 'wait_decay': 0.684772386716657, 'uncertainty_gate_threshold': 0.21938207130919019, 'slack_pressure_gate_steepness': 1.2374184628494664, 'remaining_work_weight': 0.18102330956850315, 'energy_uncertainty_interaction': 0.1725020477905767, 'ddl_protection_gate_slope': 3.4626881295056697}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '85f3c1fd4542bafc2c57003cfa4b83cf8bced63ee945abca27f433d3635cdd68', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's robust median-MAD normalization and validated slack-pressure gates 
       with Parent 1's slack-aware uncertainty gating and dedicated starvation relief under pressure.
       Structural change: replaces fixed uncertainty_gate_threshold with adaptive variant scaled by sigmoid(slack),
       and unifies starvation relief into a single pressure-gated linear term — removing 'wait_saturation_offset'
       since logistic saturation was dropped in favor of bounded linear decay."""
    eps = 0.004927901068561355
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
    ddl_gate = 1.0 / (1.0 + np.exp(-3.4626881295056697 * slack))
    adaptive_uncert_thresh = 0.21938207130919019 * (1.0 - 1.0 / (1.0 + np.exp(-slack)))
    slack_aware_uncert_gate = (norm_uncert > adaptive_uncert_thresh).astype(float) * ddl_gate
    ddl_breach = (slack <= 0.0).astype(float)
    critical_path_leverage = norm_rank * norm_work * ddl_breach
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.726339499843255
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    coupled_slack = np.clip(norm_slack, -1.0, 1.0)
    rank_slack_coupling = 1.0 + 1.8467165687691889 * coupled_slack
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-1.2374184628494664 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * rank_slack_coupling * (1.0 + 1.8467165687691889 * rank_gate)
    duration_risk_score = norm_duration * slack_aware_uncert_gate * slack_pressure
    wait_benefit_pressure = np.clip(0.684772386716657 * ready_wait_time, 0.0, 2.0) * ddl_breach
    energy_uncert_penalty = norm_energy * norm_uncert * slack_aware_uncert_gate
    energy_slack_penalty = norm_energy * (1.0 + 2.726339499843255 * slack_pressure) * ddl_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 1.8375025150459485 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit_pressure, -2.0, 2.0) + 0.9432786612092171 * np.clip(duration_risk_score, -2.0, 2.0) + 0.1725020477905767 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 2.726339499843255 * np.clip(energy_slack_penalty, -2.0, 2.0) + 0.18102330956850315 * np.clip(norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
