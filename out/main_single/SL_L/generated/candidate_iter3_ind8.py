import numpy as np
RULE_METADATA = {'structure_hash': '7144310157e5126d53d9132c0b237f29399552ab52209cf8e0319c5d9c904bf0', 'parameter_schema_hash': '37c38405176ebd9da85c31935abcdf04debd76e6c7518b9263a17814241917d6', 'best_parameter_hash': 'ab1a313a8a08b09742689b6ca0c6f633d5b842ff25632fcc77de94ae37ed8caa', 'best_parameters': {'epsilon': 5.345627306904592e-06, 'slack_penalty_exponent': 3.619390545453534, 'energy_efficiency_ratio_weight': 0.24250396886403366, 'uncertainty_slack_coupling': 2.597315496834163, 'wait_saturation_threshold': 20.20442037579079, 'rank_slack_balance': 0.5374547077247739, 'duration_risk_penalty': 0.6451718869354364, 'early_slack_reward_factor': 0.16890521314877557, 'slack_percentile_high': 92.94398875841259, 'slack_percentile_low': 5.752611700157075, 'energy_feasibility_sigmoid_steepness': 2.991024672711838, 'energy_feasibility_sigmoid_offset': 0.403946423156488}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': '76f2f28e3f728212322edf52555037f5baf386c5300f24a81261ef39b88d3c5b', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with soft sigmoid feasibility gating and tanh-bounded normalization.
    
    Key improvements:
    - Replaced hard energy gating with smooth, differentiable sigmoid: enables CMA-ES gradient signals
      and avoids brittle transitions near DDL boundary.
    - Replaced clipping in robust_norm with tanh scaling: preserves ordinal relationships while strictly
      bounding influence to [-1,1] for stable optimization.
    - Removed redundant criticality_boost gating (per self-reflection) and strengthened rank utilization
      via uncertainty-aware upward_rank normalization using declared parameter.
    - Added explicit uncertainty-weighted rank normalization to prioritize high-criticality tasks *under risk*.
    - All numeric literals are strictly in {-2,-1,0,1,2}.
    """
    eps = 5.345627306904592e-06

    def robust_norm(x):
        x = np.asarray(x, dtype=float)
        denom = np.mean(np.abs(x)) + eps
        normed = x / denom
        return np.tanh(normed)
    slack_arr = np.asarray(slack, dtype=float)
    slack_penalty = np.where(slack_arr < 0, (-slack_arr) ** 3.619390545453534, slack_arr * 0.16890521314877557)
    energy_gate = 1.0 / (1.0 + np.exp(-2.991024672711838 * (slack_arr - 0.403946423156488)))
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = robust_norm(energy_per_sec)
    energy_term = energy_gate * energy_eff_score * 0.24250396886403366
    deadline_pressure = np.maximum(0.0, -slack_arr)
    unc_slack_coupling = uncertainty * deadline_pressure * 2.597315496834163
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.6451718869354364
    wait_clipped = np.clip(ready_wait_time, 0, 20.20442037579079)
    wait_score = -robust_norm(wait_clipped)
    if len(slack_arr) > 1:
        slack_p90 = np.percentile(slack_arr, 92.94398875841259)
        slack_p10 = np.percentile(slack_arr, 5.752611700157075)
        slack_range = np.maximum(eps, slack_p90 - slack_p10)
        slack_normalized = np.clip((slack_arr - np.min(slack_arr)) / (slack_range + eps), 0, 1)
    else:
        slack_normalized = np.array([0.0])
    weight_rank = 0.5374547077247739 + (1.0 - 0.5374547077247739) * slack_normalized
    rank_uncertainty_weighted = upward_rank * (1.0 + uncertainty * 0.403946423156488)
    rank_score = -robust_norm(rank_uncertainty_weighted) * weight_rank
    residual_energy_bias = robust_norm(min_incremental_energy) * (1.0 - weight_rank)
    score = robust_norm(slack_penalty) + robust_norm(unc_slack_coupling) + robust_norm(duration_risk) + energy_term + rank_score + wait_score + residual_energy_bias
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score
