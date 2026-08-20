import numpy as np
RULE_METADATA = {'structure_hash': 'd43d05540ad80575a9890b1215ade0f6e83cd77b3ea821d4005c6b2f31af63c8', 'parameter_schema_hash': '4203b476a7db709c5ee4de94ace5536100dc3447db3eece05f858f018b3fc080', 'best_parameter_hash': '479c12cf6a8f416177726ef0d6ba8d6d92d4b85ae12c7d18db7f61cbada7318e', 'best_parameters': {'epsilon': 1.4250297854582603e-06, 'criticality_scale': 0.5447900189945063, 'energy_sensitivity': 0.1793295678529872, 'energy_uncertainty_interaction': 0.7433768796557942, 'remaining_work_weight': 0.40277284195252705, 'ddl_protection_gate_slope': 3.4279577808622452, 'uncertainty_gate_threshold': 0.22712874514920772, 'wait_decay': 0.016807569357916368, 'successor_release_exponent': 0.8014172740197216, 'slack_pressure_gate_center': 0.4057223694651103, 'slack_pressure_gate_halfwidth': 0.20111247663585347}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': 'bc7fad573797518cc920708d4b8a547a6cb5bfd4893edcf75bd72d0f0f6d751c', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: merges Parent 2's numerical stability with Parent 1's robust successor-release modeling;
       introduces exponentiated successor-release interaction (validated by DDL diagnostics) for stronger critical-path unblocking;
       replaces linear DDL gate with centered linear ramp for sharper, tunable feasibility enforcement;
       retains clipped linear slack penalty and median-MAD normalization for monotonicity and outlier resilience."""
    eps = 1.4250297854582603e-06
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
    gate_lower = 0.4057223694651103 - 0.20111247663585347
    gate_upper = 0.4057223694651103 + 0.20111247663585347
    ddl_pressure = np.clip((norm_slack - gate_lower) / (gate_upper - gate_lower + eps), 0.0, 1.0)
    ddl_pressure = np.where(norm_slack <= gate_lower, 0.0, ddl_pressure)
    ddl_pressure = np.where(norm_slack >= gate_upper, 1.0, ddl_pressure)
    successor_release = (norm_rank * norm_work) ** 0.8014172740197216 * ddl_pressure
    raw_slack_penalty = np.maximum(-slack, 0.0)
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    coupled_rank = norm_rank * (1.0 + 0.5447900189945063 * ddl_pressure)
    uncert_gate = np.where(norm_uncert > 0.22712874514920772, 1.0, 0.0)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_pressure
    wait_benefit = np.clip(0.016807569357916368 * ready_wait_time, 0.0, 1.0)
    exec_penalty = norm_duration * ddl_pressure
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - np.clip(coupled_rank, -2.0, 2.0) - 0.1793295678529872 * np.clip(norm_energy * ddl_pressure, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(exec_penalty, -2.0, 2.0) + 0.7433768796557942 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 0.40277284195252705 * np.clip(norm_work, -2.0, 2.0) + np.clip(ddl_pressure * 3.4279577808622452, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
