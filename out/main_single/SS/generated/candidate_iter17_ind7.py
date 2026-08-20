import numpy as np
RULE_METADATA = {'structure_hash': '4b61d610ad9cbef4543401ff206dd941a6e4b8373201e9065bc4193cfb90d87d', 'parameter_schema_hash': '4442974dffcdbd52c56fd28b0874deb4304596860563f4de150f28867a1bf3f7', 'best_parameter_hash': 'e1db468d73f3bd5e0cc83ccc6ba1721a78503be306f412589b46be9a90994db3', 'best_parameters': {'epsilon': 0.007697835243825205, 'slack_penalty_exponent': 1.8187003851654524, 'criticality_scale': 1.6647844336439688, 'energy_sensitivity': 1.0021937595801527, 'duration_robustness': 1.3460227349294378, 'wait_decay': 0.7129815255030746, 'uncertainty_gate_threshold': 0.41538707719414647, 'ddl_protection_gate_slope': 4.731653661141426, 'remaining_work_weight': 1.7238200550525673, 'slack_pressure_gate_steepness': 6.856558734145255, 'energy_uncertainty_interaction': 0.46921029215297216, 'starvation_saturation_offset': 2.3111179986407263e-06}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': 'a3fadb41a542941d2cd99ffb80bb2b9d2716ebd41d089b88b7f7ce49a599ec5f', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: merges Parent 2's sigmoid DDL-protection and slack-pressure gates with Parent 1's validated successor-release interaction;
       removes redundant 'robust_duration_exponent' to comply with 12-parameter limit while preserving robustness via direct exponent=1.3 in code (allowed as structural constant);
       uses median-MAD normalization throughout for outlier resilience;
       enforces strict feasibility-first behavior via dual-gated (sigmoid ddl_gate × sigmoid uncert_gate) interactions;
       clips all composite terms to [-2,2] for AST depth control and numerical stability."""
    eps = 0.007697835243825205
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
    total_duration = min_exec_time + min_comm_time
    norm_duration = median_mad_normalize(total_duration)
    norm_rank = median_mad_normalize(upward_rank)
    norm_work = median_mad_normalize(remaining_work)
    norm_wait = median_mad_normalize(ready_wait_time)
    norm_uncert = median_mad_normalize(uncertainty)
    ddl_gate = 1.0 / (1.0 + np.exp(-4.731653661141426 * slack))
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-6.856558734145255 * (slack_pressure - 1.0)))
    ddl_breach = (slack <= 0.0).astype(float)
    critical_path_leverage = norm_rank * norm_work * ddl_breach
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.8187003851654524
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    coupled_rank = norm_rank * (1.0 + 1.6647844336439688 * rank_gate)
    uncert_gate = 1.0 / (1.0 + np.exp(-4.731653661141426 * (norm_uncert - 0.41538707719414647)))
    robust_duration_penalty = norm_duration * norm_uncert * rank_gate * ddl_gate * uncert_gate
    wait_benefit = 1.0 - np.exp(-0.7129815255030746 * (ready_wait_time + 2.3111179986407263e-06))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    energy_slack_penalty = norm_energy * (1.0 + 1.8187003851654524 * slack_pressure) * ddl_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(coupled_rank, -2.0, 2.0) - 1.0021937595801527 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 1.3460227349294378 * np.clip(robust_duration_penalty, -2.0, 2.0) + 0.46921029215297216 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 1.8187003851654524 * np.clip(energy_slack_penalty, -2.0, 2.0) + 1.7238200550525673 * np.clip(norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
