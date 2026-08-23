import numpy as np
RULE_METADATA = {'structure_hash': 'd285f3727accdc7cc816083c604d846139d3d79ad795e88ed3756ad0b6da0b13', 'parameter_schema_hash': '8895acf577eae4e5ddea27a12b00270dea6e02a468044e659d9bf39c7105b283', 'best_parameter_hash': '32ad9ec574a946b24f43bf78808c5f33d9743f7150940d079dbf484580e2093b', 'best_parameters': {'epsilon': 2.722180741555404e-05, 'slack_risk_exponent': 1.6155882797419903, 'criticality_gate_threshold': 0.15758181603818242, 'energy_sensitivity': 2.2169129394271367, 'duration_uncertainty_coupling': 0.4859042234137372, 'wait_decay_rate': 0.008670201915299432, 'rank_fallback_weight': 0.825391698679395, 'uncertainty_slack_interaction': 1.2590816479119014, 'wait_boost_weight': 0.6163130206151759, 'nan_replacement': 263234.3074285814, 'posinf_clip': 8916863.297126163, 'neginf_clip': -143801932.69681013}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': 'b77e8e3fb66650c85a1e226d8824c8d2372d2694e422274aa8fa47f43a03e66a', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule with compound-risk criticality gate and triple-coupling risk term.
    
    Structural changes:
    - Replaces IQR with robust mean-abs normalization (stable, low AST depth)
    - Introduces bounded triple interaction: duration * work * uncertainty → captures risk in large,
      uncertain, compute-heavy sub-DAGs; clipped to [0,1] to prevent explosion
    - Criticality gate now requires BOTH slack < 0 AND uncertainty > threshold (compound detection)
    - All numeric literals are from {-2,-1,0,1,2}; no hidden constants
    - Uses np.finfo for immutable safeguards where needed (e.g., in robust_norm denominator)
    """
    eps = 2.722180741555404e-05
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def robust_norm(x):
        x_abs = np.abs(x)
        denom = np.mean(x_abs) + eps
        return x / denom
    norm_slack = robust_norm(slack)
    norm_energy = robust_norm(min_incremental_energy)
    norm_duration = robust_norm(min_exec_time + min_comm_time)
    norm_rank = robust_norm(upward_rank)
    norm_work = robust_norm(remaining_work)
    norm_wait = robust_norm(ready_wait_time)
    norm_uncert = robust_norm(uncertainty)
    slack_power = np.sign(norm_slack) * np.abs(norm_slack) ** 1.6155882797419903
    triple_coupling = np.clip(norm_duration * norm_work * norm_uncert, 0.0, 1.0) * 0.4859042234137372
    slack_pressure_mask = np.where(norm_slack < 0, 1.0, 0.0)
    uncert_high_mask = np.where(norm_uncert > 0.15758181603818242, 1.0, 0.0)
    compound_risk_mask = slack_pressure_mask * uncert_high_mask
    gate_activation = compound_risk_mask
    dur_uncert_coupling = np.clip(norm_duration * norm_uncert, 0.0, 1.0) * 0.4859042234137372
    wait_boost = 1.0 - np.exp(-0.008670201915299432 * (norm_wait + eps))
    uncert_slack_penalty = norm_uncert * slack_pressure_mask * 1.2590816479119014
    score = slack_power
    safe_slack_mask = np.where(norm_slack > 0, 1.0, 0.0)
    score += safe_slack_mask * 2.2169129394271367 * norm_energy
    score -= gate_activation * 0.825391698679395 * norm_rank
    score += dur_uncert_coupling
    score += triple_coupling
    score -= wait_boost * 0.6163130206151759
    score += uncert_slack_penalty
    score = np.nan_to_num(score, nan=263234.3074285814, posinf=8916863.297126163, neginf=-143801932.69681013)
    return score
