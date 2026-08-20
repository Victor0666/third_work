import numpy as np
RULE_METADATA = {'structure_hash': 'c3c1fcbf4ebab9380d1be6be9bff7c6246b0740014c632363f5f398951474ef0', 'parameter_schema_hash': '609b0ce7b69bc976a7a01ed23e7c6b8e0a7abff6ebb36357b2e766778f108352', 'best_parameter_hash': '351c02bc7e0107c8f612ea3a13f73df407ffa18624881f50f60bee7a153e697b', 'best_parameters': {'epsilon': 0.02002388163429561, 'slack_penalty_exponent': 1.360218922617832, 'criticality_scale': 1.3559271551236836, 'energy_sensitivity': 0.15436536856117236, 'duration_robustness': 1.7143277429044785, 'wait_decay': 0.4957325434845682, 'uncertainty_gate_threshold': 0.025420891289830837, 'slack_gate_width': 0.22616820080639363, 'slack_sensitivity': 1.1927200032359822, 'energy_uncertainty_interaction': 0.35210253204813047, 'work_leverage_factor': 0.6115854364097524}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': 'e57d895c52330eada456f4a87392b26f78b6c41982b9e54f9bc4c16a525b11c2', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: inherits robust sign-preserving normalization and unified slack pressure from Parent 2;
       adds energy-uncertainty interaction and slack-gated work leverage from Parent 1;
       removes harmful work_pressure mask; uses linear slack reward + power-law penalty for balanced feasibility-first ranking;
       all literals are -2,-1,0,1,2; epsilon via PARAMS; no side effects."""
    eps = 0.02002388163429561
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
        center = np.median(x)
        scale = np.median(np.abs(x - center)) + eps
        return (x - center) / scale
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    slack_pressure = np.power(np.clip(-norm_slack, 0.0, None), 1.360218922617832)
    rank_gate = np.clip(1.0 - 2.0 * np.clip(norm_slack, 0.0, 0.22616820080639363), 0.0, 1.0)
    boosted_rank = norm_rank * (1.0 + 1.3559271551236836 * rank_gate)
    uncert_gate = ((norm_uncert > 0.025420891289830837) & (slack_pressure > 0)).astype(float)
    duration_risk_interaction = norm_duration * uncert_gate * 1.7143277429044785
    energy_uncert_gate = ((norm_uncert > 0.025420891289830837) & (slack_pressure > 0)).astype(float)
    energy_uncert_penalty = norm_energy * norm_uncert * energy_uncert_gate * 0.35210253204813047
    wait_benefit = np.tanh(0.4957325434845682 * norm_wait)
    slack_reward = 1.1927200032359822 * np.clip(norm_slack, None, 0.0)
    work_leverage_gate = (slack >= 0.0).astype(float)
    work_leverage = norm_work * work_leverage_gate * 0.6115854364097524
    score = +slack_pressure + 0.15436536856117236 * norm_energy - boosted_rank + duration_risk_interaction - wait_benefit - slack_reward + energy_uncert_penalty - work_leverage
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
