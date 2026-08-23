import numpy as np
RULE_METADATA = {'structure_hash': 'c795b68e5ec797e512324d92bad73e7656245541b7cc496fe1ad79646aa197b2', 'parameter_schema_hash': 'b7654b34af74cf7d09ef4782e774b530ad87dc5f7ce02bc089fbc8735c143eab', 'best_parameter_hash': '4c35abc97528305733fbf525086bd070babf8c13fe2c86e09824e4064592a517', 'best_parameters': {'epsilon': 0.0029686869544096792, 'slack_risk_exponent': 3.263380178639596, 'criticality_gate_threshold': 0.20137847108007, 'energy_sensitivity': 1.3241328097575598, 'duration_uncertainty_coupling': 0.0318914933792915, 'wait_decay_rate': 0.06919441044144348, 'rank_fallback_weight': 1.409039465612329, 'uncertainty_slack_interaction': 0.6012108942149819, 'wait_boost_weight': 0.45468372966729387, 'nan_replacement': -164385.46699633356, 'posinf_clip': 4054547.3878410086, 'neginf_clip': -69541249.97236502}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': '632e666d5b7c6c9dd0d3915732f67f84bb2713d62395317ad5efcf3fe77afc11', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: combines Parent 2's smooth gating and exponential wait boost with Parent 1's energy-efficiency ratio.
    Smaller score = higher priority. All numeric literals are -2,-1,0,1,2 or derived from PARAMS."""
    eps = 0.0029686869544096792
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
    norm_wait = robust_norm(ready_wait_time)
    norm_uncert = robust_norm(uncertainty)
    slack_power = np.sign(norm_slack) * np.abs(norm_slack) ** 3.263380178639596
    slack_normalized_for_gate = (norm_slack - 0.20137847108007) / (1.0 + eps)
    gate_activation = np.clip(-slack_normalized_for_gate, 0.0, 1.0)
    dur_uncert_coupling = np.clip(norm_duration * norm_uncert, 0.0, 1.0) * 0.0318914933792915
    wait_boost = 1.0 - np.exp(-0.06919441044144348 * (norm_wait + eps))
    slack_pressure_mask = np.where(norm_slack < 0, 1.0, 0.0)
    uncert_slack_penalty = norm_uncert * slack_pressure_mask * 0.6012108942149819
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = robust_norm(energy_per_sec)
    score = slack_power
    safe_slack_mask = np.where(norm_slack > 0, 1.0, 0.0)
    score += safe_slack_mask * 1.3241328097575598 * norm_energy
    score += gate_activation * 1.409039465612329 * -norm_rank
    score += dur_uncert_coupling
    score -= wait_boost * 0.45468372966729387
    score += uncert_slack_penalty
    score += safe_slack_mask * energy_eff_score
    score = np.nan_to_num(score, nan=-164385.46699633356, posinf=4054547.3878410086, neginf=-69541249.97236502)
    return score
