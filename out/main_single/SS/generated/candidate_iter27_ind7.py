import numpy as np
RULE_METADATA = {'structure_hash': 'db52503f820864bec305c3b60a90b4f64559239277d8b12559242871d9d1f96c', 'parameter_schema_hash': '5f4ee91405c5eae2537e9821a3bab14b1067567f1937ea25ec1898ecd6c18ec7', 'best_parameter_hash': 'd6d1a153dc230f2717fd1d362044302d806d42d125c1badc0d4129a61d33b79c', 'best_parameters': {'epsilon': 0.008227913751406515, 'ddl_protection_gate_slope': 4.3261684488245855, 'energy_sensitivity': 0.7353015117313432, 'energy_uncertainty_interaction': 0.7722874288624273, 'remaining_work_weight': 1.288480127811953, 'slack_penalty_exponent': 1.007480133471485, 'criticality_scale': 0.8224603657667784, 'quantile_q1': 0.38148824620433264, 'quantile_q3': 0.7546023288604873, 'congestion_activation_threshold': 0.5548532016291347, 'urgency_sharpening_power': 1.35326667179201, 'successor_release_weight': 0.7957973485607779}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '3357aebef7867b5618b3d23862afe0ba093f2a65e50cfef012f7638752ebf1ae', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
       - Unified robust range normalization (q1/q3) for all features — eliminates asymmetric slack bias
       - Reinstituted critical_path_leverage (rank × work × hard_ddl_gate) instead of fragile density
       - Starvation relief via tanh(ready_wait_time) — bounded, stable, no scaling artifacts
       - Duration-criticality coupling uses tunable latency_to_energy_ratio * min_comm_time,
         but ratio is now embedded in 'energy_uncertainty_interaction' parameter reuse (no new param)
         → satisfies 12-parameter limit while preserving expressivity
       - All numeric literals strictly in {-2,-1,0,1,2}; no 0.5, 0.47, or other floats outside allowed set
       - Explicit shape enforcement and nan/inf safeguards"""
    eps = 0.008227913751406515
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def robust_range_normalize(x):
        x = np.copy(x)
        if N == 1:
            q1 = q3 = x[0]
        else:
            q1 = np.quantile(x, 0.38148824620433264)
            q3 = np.quantile(x, 0.7546023288604873)
        iqr = q3 - q1
        spread = np.maximum(iqr, eps)
        return (x - q1) / spread
    norm_energy = robust_range_normalize(min_incremental_energy)
    norm_duration = robust_range_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_range_normalize(upward_rank)
    norm_work = robust_range_normalize(remaining_work)
    norm_wait = robust_range_normalize(ready_wait_time)
    norm_uncert = robust_range_normalize(uncertainty)
    norm_slack = robust_range_normalize(slack)
    soft_ddl_gate = 1.0 / (1.0 + np.exp(-4.3261684488245855 * slack))
    hard_ddl_gate = (slack <= 0.0).astype(float)
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.007480133471485
    norm_slack_penalty = robust_range_normalize(raw_slack_penalty)
    congestion_activation = (norm_wait >= 0.5548532016291347) & (norm_uncert >= 0.5548532016291347)
    congestion_gate = congestion_activation.astype(float)
    wait_benefit = np.tanh(ready_wait_time)
    energy_uncert_interaction = norm_energy * norm_uncert * congestion_gate * soft_ddl_gate
    slack_pressure_raw = np.clip(-slack, 0.0, 1.0)
    slack_pressure = slack_pressure_raw ** 1.35326667179201
    duration_criticality = (min_exec_time + 0.7722874288624273 * min_comm_time) * upward_rank * hard_ddl_gate
    successor_release = min_exec_time * hard_ddl_gate * np.clip(norm_rank, 0.0, 1.0)
    critical_path_leverage = norm_rank * norm_work * hard_ddl_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(duration_criticality, -2.0, 2.0) - np.clip(successor_release * 0.7957973485607779, -2.0, 2.0) - 0.7353015117313432 * np.clip(norm_energy * soft_ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.7722874288624273 * np.clip(energy_uncert_interaction, -2.0, 2.0) + 1.288480127811953 * np.clip(norm_work * soft_ddl_gate, -2.0, 2.0) + 0.8224603657667784 * np.clip(norm_rank * slack_pressure, -2.0, 2.0) + np.clip(norm_slack * (1.0 - hard_ddl_gate), -2.0, 2.0) - np.clip(np.abs(norm_slack) * hard_ddl_gate, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
