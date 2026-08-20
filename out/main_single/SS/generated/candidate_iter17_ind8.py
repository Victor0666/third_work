import numpy as np
RULE_METADATA = {'structure_hash': '4526e1217168f8d6cebaa14e6572a2d9c86f9d4b871a297f4164d529eefe9b70', 'parameter_schema_hash': '3f20d25417120d9ce1f82f43455404330b33c8e9b00dd1d314ed85d9851caa6c', 'best_parameter_hash': '1535fc795ef01b4fb63151ad0a5d185a530e36eb7af11e729a59845909c2d237', 'best_parameters': {'epsilon': 0.009254986253529641, 'slack_penalty_exponent': 1.9518264796574702, 'energy_sensitivity': 1.3609331411864776, 'duration_robustness': 0.8984011443562255, 'wait_decay': 0.08914822929620775, 'uncertainty_gate_threshold': 0.27919645713118857, 'slack_pressure_gate_steepness': 1.452312980537405, 'remaining_work_weight': 0.7001951929500534, 'wait_saturation_offset': 1.196945582676493e-05, 'energy_uncertainty_interaction': 0.7759130344835433, 'ddl_protection_gate_slope': 3.66279223397662}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': 'b8b3d23c4d4f5ae3cdcabf10e5a8dcafe45225e2c4ac40c73445c1990da11e6f', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: merges Parent 2's DDL-protection gate and robust normalization with Parent 1's smooth energy ramp;
       introduces *dual-gated energy activation*: both DDL-feasibility (smooth Heaviside) AND marginal slack alignment (linear ramp);
       replaces brittle critical-path leverage with *normalized successor-unblocking potential* — product of upward_rank and normalized remaining_work,
       gated by slack pressure only (not just breach), improving upstream prioritization under tight-but-feasible deadlines;
       uses median-MAD normalization throughout for outlier resilience;
       all terms clipped to [-2,2] for stability and bounded AST depth."""
    eps = 0.009254986253529641
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
    ddl_gate = 1.0 / (1.0 + np.exp(-3.66279223397662 * slack))
    energy_ramp_center = 0.0
    energy_ramp_width = 1.0 / (3.66279223397662 + eps)
    energy_ramp = np.clip((slack - energy_ramp_center) / (energy_ramp_width + eps), 0.0, 1.0)
    energy_activation = ddl_gate * energy_ramp
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-1.452312980537405 * (slack_pressure - 1.0)))
    unblocking_potential = norm_rank * norm_work * rank_gate
    uncert_gate = 1.0 / (1.0 + np.exp(-3.66279223397662 * (norm_uncert - 0.27919645713118857)))
    duration_risk_score = norm_duration * norm_uncert * rank_gate * ddl_gate
    wait_benefit = 1.0 - np.exp(-0.08914822929620775 * (ready_wait_time + 1.196945582676493e-05))
    energy_uncert_penalty = norm_energy * norm_uncert * energy_activation
    energy_slack_penalty = norm_energy * (1.0 + 1.9518264796574702 * slack_pressure) * energy_activation
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.9518264796574702
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(unblocking_potential, -2.0, 2.0) - np.clip(1.3609331411864776 * norm_energy * energy_activation, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.8984011443562255 * np.clip(duration_risk_score, -2.0, 2.0) + 0.7759130344835433 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 1.9518264796574702 * np.clip(energy_slack_penalty, -2.0, 2.0) + 0.7001951929500534 * np.clip(norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
