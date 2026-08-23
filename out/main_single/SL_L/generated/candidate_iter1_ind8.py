import numpy as np
RULE_METADATA = {'structure_hash': '731cf62443f076ba1842962ec82d143d8c266654a2d99a9bb7b56bfdfe950e01', 'parameter_schema_hash': '85b10d31a2ce1d0ed8819d8eaf50883a0eedcd1ab855798b5edd90666ae9c1e6', 'best_parameter_hash': '70e079ac4f08c19b6e4e89e8a112db03969a976f252a2862cdee9db87d613d43', 'best_parameters': {'epsilon': 5.231935414575213e-05, 'slack_penalty_exponent': 2.629903296937988, 'criticality_boost': 1.1188511541856065, 'energy_efficiency_ratio_weight': 0.10001823834399436, 'uncertainty_slack_coupling': 0.35538883427246026, 'wait_saturation_threshold': 29.31258298780096, 'rank_slack_balance': 0.8205013365925365, 'duration_risk_penalty': 0.55157232794071, 'early_slack_reward_factor': 0.6167554523039913, 'slack_percentile_high': 98.79120004868716, 'slack_percentile_low': 6.018101529662694}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': '5e5992d4f6629f163f5198fb23112750ee5f01a6058ce25b828e9ce35a0e2951', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule: all tunable constants declared; only -2,-1,0,1,2 used as literals."""
    eps = 5.231935414575213e-05

    def robust_norm(x):
        x = np.asarray(x, dtype=float)
        denom = np.mean(np.abs(x)) + eps
        return x / denom
    slack_signed = np.copy(slack)
    slack_abs = np.abs(slack_signed)
    slack_sign = np.sign(slack_signed)
    slack_score = np.where(slack_signed < 0, slack_abs ** 2.629903296937988, slack_signed * 0.6167554523039913)
    rank_median = np.median(upward_rank) if len(upward_rank) > 1 else np.mean(upward_rank)
    is_high_rank = upward_rank >= rank_median
    is_tight_slack = slack_signed <= 2 * eps
    critical_gate = np.where(is_high_rank & is_tight_slack, 1.1188511541856065, 1.0)
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = robust_norm(energy_per_sec)
    deadline_pressure = np.maximum(0.0, -slack_signed)
    unc_slack_coupling = uncertainty * deadline_pressure * 0.35538883427246026
    wait_clipped = np.clip(ready_wait_time, 0, 29.31258298780096)
    wait_score = -robust_norm(wait_clipped)
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.55157232794071
    slack_p90 = np.percentile(slack, 98.79120004868716) if len(slack) > 1 else np.max(slack)
    slack_p10 = np.percentile(slack, 6.018101529662694) if len(slack) > 1 else np.min(slack)
    slack_range = np.maximum(eps, slack_p90 - slack_p10)
    slack_normalized = np.clip((slack_signed - np.min(slack)) / (slack_range + eps), 0, 1)
    weight_rank = 0.8205013365925365 + (1 - 0.8205013365925365) * slack_normalized
    rank_score = -robust_norm(upward_rank) * weight_rank
    score = robust_norm(slack_score) + robust_norm(unc_slack_coupling) + robust_norm(duration_risk) + 0.10001823834399436 * energy_eff_score + rank_score + wait_score + robust_norm(min_incremental_energy) * (1 - weight_rank)
    score = score * critical_gate
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score
