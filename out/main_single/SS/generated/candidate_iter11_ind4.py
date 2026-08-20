import numpy as np
RULE_METADATA = {'structure_hash': '9cd25f69c9b8ea7e6126a224a9262632e5091497b084385826e7226f5c360621', 'parameter_schema_hash': '5b3d33a88c3f20637b3ffddbba47496360a3a428f012a2ae43156bb237d7bb83', 'best_parameter_hash': '81293c4d20e5b562a3d3e62396cc55fec09b4dd99801daf843e06bc24ce9ea1b', 'best_parameters': {'epsilon': 7.627980607831276e-05, 'slack_penalty_exponent': 2.941249392607477, 'criticality_scale': 0.5407897443429523, 'energy_sensitivity': 1.1269619606370112, 'duration_robustness': 1.1518767660619256, 'wait_decay': 0.2925344970842554, 'uncertainty_gate_threshold': 0.7294588580058203, 'remaining_work_weight': 0.005518817425670997, 'wait_saturation_offset': 3.3903938839788086e-08, 'energy_uncertainty_interaction': 0.5291777742781089, 'ddl_urgency_steepness': 1.9862855143642737, 'rank_boost_floor': 0.2854568638687526}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': 'e01aab1176b691ae1804272384e05ec07ee79c0245a0238551144a37fc4e5b7b', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: merges Parent 2's smooth urgency gate and rank floor with Parent 1's successor-release interaction.
       Uses robust median-MAD normalization; all operations safeguarded against NaN/inf/zero.
       Prioritizes hard DDL feasibility first, then critical path, then energy efficiency, then starvation relief — with strict ordering via gated terms."""
    eps = 7.627980607831276e-05
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
        x = np.asarray(x, dtype=float)
        center = np.median(x) if N > 1 else x[0]
        spread = np.median(np.abs(x - center)) if N > 1 else np.abs(x[0] - center) + eps
        return (x - center) / (spread + eps)
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    median_abs_slack = np.median(np.abs(slack)) if N > 1 else np.abs(slack[0])
    urgency_input = -slack / (median_abs_slack + eps)
    ddl_urgency_gate = 1.0 / (1.0 + np.exp(-1.9862855143642737 * urgency_input))
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.941249392607477
    rank_boost = 0.2854568638687526 + (1.0 - 0.2854568638687526) * ddl_urgency_gate
    protected_rank = norm_rank * rank_boost
    successor_release_impact = norm_rank * norm_work
    wait_raw = 0.2925344970842554 * (ready_wait_time + 3.3903938839788086e-08)
    wait_benefit = np.clip(1.0 - np.exp(-wait_raw), 0.0, 1.0)
    energy_uncert_penalty = norm_energy * norm_uncert * np.where((norm_energy > 0.0) & (norm_uncert > 0.7294588580058203), 1.0, 0.0)
    duration_risk_score = norm_duration * np.where((slack <= eps) & (uncertainty >= 0.7294588580058203), 1.0, 0.0)
    score = +raw_slack_penalty - 0.5407897443429523 * protected_rank - successor_release_impact - 1.1269619606370112 * norm_energy - norm_duration - wait_benefit + 1.1518767660619256 * duration_risk_score + 0.5291777742781089 * energy_uncert_penalty + 0.005518817425670997 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
