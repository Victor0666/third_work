import numpy as np
RULE_METADATA = {'structure_hash': 'baeacf4ec81a5ade60884eadd66a1384ec7da5c0ad07f250fe2fe879f5e5b74e', 'parameter_schema_hash': 'b7654b34af74cf7d09ef4782e774b530ad87dc5f7ce02bc089fbc8735c143eab', 'best_parameter_hash': '3018f98edf48e35501175574484a7e6f4516c1905ca8683c2acce1e08ed9f116', 'best_parameters': {'epsilon': 0.0009553528023159712, 'slack_risk_exponent': 3.5484970183816107, 'criticality_gate_threshold': 0.06844647173293705, 'energy_sensitivity': 1.105050156148152, 'duration_uncertainty_coupling': 0.6646975236405153, 'wait_decay_rate': 0.07852566892198519, 'rank_fallback_weight': 0.33681084521039006, 'uncertainty_slack_interaction': 1.119504132781943, 'wait_boost_weight': 0.28311850758239315, 'nan_replacement': -496815.1262345449, 'posinf_clip': 7882009.744907906, 'neginf_clip': -455241131.87609696}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': '280ac1b4cf067a1c1d1604fbebb95b852ef05ff22f43c648b2b692067ec7578c', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining robust risk-gated criticality (Parent 2) with energy-duration efficiency.
    Removes 'remaining_work_penalty' to comply with parameter count limit; retains all other structural improvements.
    Uses smooth gate, exponential wait boost, decoupled uncertainty interactions, and adds energy_per_duration_ratio.
    All numeric literals are -2,-1,0,1,2 or derived from PARAMS."""
    eps = 0.0009553528023159712
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
    norm_duration = robust_norm(min_exec_time + min_comm_time + eps)
    norm_rank = robust_norm(upward_rank)
    norm_wait = robust_norm(ready_wait_time)
    norm_uncert = robust_norm(uncertainty)
    slack_power = np.sign(norm_slack) * np.abs(norm_slack) ** 3.5484970183816107
    slack_normalized_for_gate = (norm_slack - 0.06844647173293705) / (1.0 + eps)
    gate_activation = np.clip(-slack_normalized_for_gate, 0.0, 1.0)
    dur_uncert_coupling = np.clip(norm_duration * norm_uncert, 0.0, 1.0) * 0.6646975236405153
    wait_boost = 1.0 - np.exp(-0.07852566892198519 * (norm_wait + eps))
    slack_pressure_mask = np.where(norm_slack < 0, 1.0, 0.0)
    uncert_slack_penalty = norm_uncert * slack_pressure_mask * 1.119504132781943
    energy_per_duration = min_incremental_energy / (min_exec_time + min_comm_time + eps)
    energy_duration_ratio = robust_norm(energy_per_duration)
    score = slack_power
    safe_slack_mask = np.where(norm_slack > 0, 1.0, 0.0)
    score += safe_slack_mask * 1.105050156148152 * norm_energy
    score -= gate_activation * 0.33681084521039006 * norm_rank
    score += dur_uncert_coupling
    score -= wait_boost * 0.28311850758239315
    score += uncert_slack_penalty
    score += energy_duration_ratio
    score = np.nan_to_num(score, nan=-496815.1262345449, posinf=7882009.744907906, neginf=-455241131.87609696)
    return score
