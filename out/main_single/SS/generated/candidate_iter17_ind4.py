import numpy as np
RULE_METADATA = {'structure_hash': 'f5da1b95dcb6c3a2582208b27b8b37a39c944b7cf47c3e4fc58e38cddf5a8e87', 'parameter_schema_hash': '81a99dd56ea570b99e2d1f11fe37751174d12b55e955cebb9cf204a4ed0cc952', 'best_parameter_hash': '8b6a8ccd6fec08bf2f1396cc3b7f25136e615af323563cccc36b86c0dff37bb4', 'best_parameters': {'ddl_protection_gate_slope': 7.904322366676061, 'energy_sensitivity': 0.6475077103500685, 'energy_uncertainty_interaction': 0.29255271781090486, 'criticality_scale': 0.5033077933864162, 'remaining_work_weight': 1.84433876588772, 'slack_penalty_exponent': 1.9201920485296577, 'slack_pressure_gate_steepness': 1.9902392660388504, 'uncertainty_gate_threshold': 0.4445816803900968, 'wait_decay': 0.20644180889152508, 'rank_amplifier_lower_bound': 0.33124228991966354, 'rank_amplifier_upper_bound': 2.589229178432373, 'duration_robustness_factor': 0.9311559938114962}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': 'ad736674f07f9f38f3b0ab14ae20cbf691c640080b0275d7a703070e8c62c1ef', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: combines Parent 2's superior normalization, nonlinear slack modeling, and bounded rank amplification with Parent 1's robust duration damping and explicit DDL-gated duration penalty.
       Structural novelty: introduces *duration_robustness_factor*-damped *norm_duration* penalty fully gated by ddl_gate and slack_pressure_gate — prevents noisy duration estimates from overriding deadline feasibility while preserving urgency coupling.
       All penalty terms now share uniform ddl_gate + slack_pressure_gate composition for consistency; anti-starvation remains additive linear bonus for monotonicity and discriminability."""
    eps = np.finfo(float).tiny
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def robust_normalize(x):
        x = np.copy(x)
        if N == 1:
            center = x[0]
            spread = eps
        else:
            center = np.mean(x)
            spread = np.std(x, ddof=0)
        spread = np.where(spread > eps, spread, eps)
        return (x - center) / spread
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    ddl_gate = 1.0 / (1.0 + np.exp(-7.904322366676061 * slack))
    raw_slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    slack_pressure_gate = 1.0 / (1.0 + np.exp(-1.9902392660388504 * (raw_slack_pressure - 1.0)))
    ddl_breach = (slack <= 0.0).astype(float)
    critical_path_score = norm_rank * norm_work * ddl_breach
    rank_amplifier = 1.0 + 0.5033077933864162 * slack_pressure_gate
    rank_amplifier = np.clip(rank_amplifier, 0.33124228991966354, 2.589229178432373)
    boosted_rank = norm_rank * rank_amplifier
    uncert_gate = 1.0 / (1.0 + np.exp(-7.904322366676061 * (norm_uncert - 0.4445816803900968)))
    comm_uncert_penalty = norm_duration * norm_uncert * uncert_gate * ddl_breach
    wait_bonus = 0.20644180889152508 * norm_wait
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    energy_slack_penalty = norm_energy * raw_slack_pressure * ddl_gate
    duration_penalty = 0.9311559938114962 * norm_duration * slack_pressure_gate * ddl_gate
    score = +np.clip(norm_slack, -2.0, 2.0) + np.clip(raw_slack_pressure * 1.9201920485296577, -2.0, 2.0) - np.clip(critical_path_score, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 0.6475077103500685 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) + np.clip(wait_bonus, -2.0, 2.0) + 0.29255271781090486 * np.clip(energy_uncert_penalty, -2.0, 2.0) + np.clip(comm_uncert_penalty, -2.0, 2.0) + 1.84433876588772 * np.clip(norm_work, -2.0, 2.0) + np.clip(duration_penalty, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
