import numpy as np
RULE_METADATA = {'structure_hash': '244732c062a5a29ba0ec32cbf1ef47a16b253f6927ef79b85d1263166cf503c5', 'parameter_schema_hash': '7d9d1c6e390170c4611a086b9524c245c3446bff92696314656e1cbff5f9ca69', 'best_parameter_hash': 'd040ab27a3bdade898f0eb2df78cdb70a90b9cfdbd058dd43b9313c9e3164392', 'best_parameters': {'epsilon': 1.3342125102977589e-05, 'slack_penalty_exponent': 1.7503451793216223, 'criticality_scale': 1.630500543116297, 'energy_sensitivity': 1.2587734548301812, 'duration_robustness': 1.079990775125509, 'wait_decay': 0.32510498803183624, 'uncertainty_gate_threshold': 0.5206599509640137, 'slack_pressure_gate_steepness': 7.5015472157950445, 'remaining_work_weight': 1.078260783310458, 'wait_saturation_offset': 3.961947671775054e-06, 'energy_uncertainty_interaction': 0.8908094035561394, 'ddl_protection_gate_slope': 2.0880037002084006}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': 'ba64e26fb5c6d0c99ce8ab09bacd1dcd5ad48be00eafcd56578d92f5d2b80a16', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's robust median-MAD normalization and sharpened gates with Parent 1's validated successor-release interaction.
       Structural novelty: (1) successor-release now couples wait time *and* descendant criticality under DDL breach, (2) unified dual-gated slack-pressure activation,
       (3) strict [-2,2] clipping per term for stability, (4) explicit anti-starvation + anti-lateness decoupling via orthogonal terms."""
    eps = 1.3342125102977589e-05
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
    ddl_gate = 1.0 / (1.0 + np.exp(-2.0880037002084006 * slack))
    ddl_breach = (slack <= 0.0).astype(float)
    critical_path_leverage = norm_rank * norm_work * ddl_breach
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.7503451793216223
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    slack_pressure_raw = np.clip(-norm_slack, 0.0, 2.0)
    slack_pressure_gate = 1.0 / (1.0 + np.exp(-7.5015472157950445 * (slack_pressure_raw - 1.0 / 2.0)))
    uncert_gate = 1.0 / (1.0 + np.exp(-2.0880037002084006 * (norm_uncert - 0.5206599509640137)))
    duration_risk_score = norm_duration * uncert_gate * slack_pressure_raw * ddl_gate
    wait_benefit = 1.0 - np.exp(-0.32510498803183624 * (ready_wait_time + 3.961947671775054e-06))
    successor_release = norm_wait * norm_rank * ddl_breach
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    energy_slack_penalty = norm_energy * (1.0 + 1.7503451793216223 * slack_pressure_raw) * ddl_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(norm_rank * (1.0 + 1.630500543116297 * slack_pressure_gate), -2.0, 2.0) - 1.2587734548301812 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 1.079990775125509 * np.clip(duration_risk_score, -2.0, 2.0) + 0.8908094035561394 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 1.7503451793216223 * np.clip(energy_slack_penalty, -2.0, 2.0) + 1.078260783310458 * np.clip(norm_work * ddl_gate, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
