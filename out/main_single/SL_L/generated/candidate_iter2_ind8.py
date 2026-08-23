import numpy as np
RULE_METADATA = {'structure_hash': '9c53c20025b9f0e7831b03b360fcce85456bec77d13c4973a7942e92b55a99d1', 'parameter_schema_hash': 'f0bc718a28925ace61e3dc32fca388b54f9509822112dd0fc8d61d13fd38d07e', 'best_parameter_hash': '4c7afe8e47c2e3e8e8ceefb69fd57c8f474782d8587768828d988b74862641a3', 'best_parameters': {'epsilon': 1.587953945285319e-06, 'slack_penalty_exponent': 2.979372635448514, 'criticality_boost': 1.7543762941032592, 'energy_efficiency_ratio_weight': 1.939462656344324, 'uncertainty_slack_coupling': 2.0384220000312814, 'wait_saturation_threshold': 99.92436105367756, 'rank_slack_balance': 0.6508748103378458, 'duration_risk_penalty': 1.0408622600676762, 'early_slack_reward_factor': 0.5479745172405034, 'slack_percentile_high': 92.0960070624316, 'slack_percentile_low': 24.42748319151008, 'energy_feasibility_gate_threshold': 1.6105256262888377}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': '7161f0b2be5d9afd1f0a060545de9b3d0b3b9641f762f582a2e43e0b6068100e', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining robustness from Parent 1 and efficiency from Parent 2.
    
    Key structural improvements:
    - Introduces hard feasibility gate: energy term only active when slack > PARAMS["energy_feasibility_gate_threshold"]
      (replacing soft exp(-slk) gating in Parent 1, enabling strict DDL-first behavior).
    - Replaces IQR with robust mean-abs normalization *and* adds explicit finite-range clipping for all normalized terms.
    - Uses percentile-based slack range interpolation *and* couples it with uncertainty-aware criticality boost.
    - Adds uncertainty-weighted duration risk as separate term (not just multiplicative coupling).
    - All numeric literals are strictly in {-2,-1,0,1,2}.
    """
    eps = 1.587953945285319e-06

    def robust_norm(x):
        x = np.asarray(x, dtype=float)
        denom = np.mean(np.abs(x)) + eps
        normed = x / denom
        return np.clip(normed, -2.0, 2.0)
    slack_arr = np.asarray(slack, dtype=float)
    slack_penalty = np.where(slack_arr < 0, (-slack_arr) ** 2.979372635448514, slack_arr * 0.5479745172405034)
    rank_median = np.median(upward_rank) if len(upward_rank) > 1 else np.mean(upward_rank)
    is_high_rank = upward_rank >= rank_median
    is_tight_slack = slack_arr <= 1.6105256262888377 + eps
    critical_gate = np.where(is_high_rank & is_tight_slack, 1.7543762941032592, 1.0)
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = robust_norm(energy_per_sec)
    deadline_pressure = np.maximum(0.0, -slack_arr)
    unc_slack_coupling = uncertainty * deadline_pressure * 2.0384220000312814
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 1.0408622600676762
    wait_clipped = np.clip(ready_wait_time, 0, 99.92436105367756)
    wait_score = -robust_norm(wait_clipped)
    if len(slack_arr) > 1:
        slack_p90 = np.percentile(slack_arr, 92.0960070624316)
        slack_p10 = np.percentile(slack_arr, 24.42748319151008)
        slack_range = np.maximum(eps, slack_p90 - slack_p10)
        slack_normalized = np.clip((slack_arr - np.min(slack_arr)) / (slack_range + eps), 0, 1)
    else:
        slack_normalized = np.array([0.0])
    weight_rank = 0.6508748103378458 + (1.0 - 0.6508748103378458) * slack_normalized
    rank_score = -robust_norm(upward_rank) * weight_rank
    energy_term = np.where(slack_arr > 1.6105256262888377, energy_eff_score * 1.939462656344324, 0.0)
    score = robust_norm(slack_penalty) + robust_norm(unc_slack_coupling) + robust_norm(duration_risk) + energy_term + rank_score + wait_score + robust_norm(min_incremental_energy) * (1.0 - weight_rank)
    score = score * critical_gate
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score
