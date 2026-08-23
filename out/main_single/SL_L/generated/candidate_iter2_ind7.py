import numpy as np
RULE_METADATA = {'structure_hash': 'aa907ed04af96a3631b3a5c7f441fd7981a570f593379fc79d78bf1a5d705689', 'parameter_schema_hash': '6f1e4ded9445a1919e19a65fba3ff8ff0be2f864bf5609e5ff91a2ad850bdf9c', 'best_parameter_hash': 'a7dad2c1fddf5a62e76aee34b71a841591fafee2b1a7eee0a9ed65ea3127d23b', 'best_parameters': {'epsilon': 9.062780215575413e-08, 'slack_penalty_exponent': 2.054819751383592, 'criticality_boost': 4.1596492433170535, 'energy_efficiency_ratio_weight': 0.9277778169833516, 'uncertainty_slack_coupling': 1.57957962031453, 'wait_saturation_threshold': 2.214184726521691, 'rank_slack_balance': 0.692436046739073, 'duration_risk_penalty': 1.601628006938996, 'early_slack_reward_factor': 0.37248870664343015, 'slack_percentile_high': 88.32319958131757, 'slack_percentile_low': 9.17138065692445, 'energy_uncertainty_interaction': 0.21820237862104608}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': 'ec44bf4528f6d702e80b5a77806705d0b4f51e728c7287d06c8c58fba5c34caa', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining robustness from Parent 2 with risk-aware energy-uncertainty coupling.
    Key improvements:
      - Replaces fragile IQR normalization with mean-abs + eps robust norm (stable across seeds)
      - Introduces novel energy_uncertainty_interaction: prioritizes low-energy tasks *especially* when uncertainty is high
      - Uses wait saturation (clipped + robust norm) instead of exponential decay to avoid numerical instability
      - Unified signed slack handling: power penalty for negative, linear reward for positive
      - Criticality boost gated only when both rank is high AND slack is non-positive (tight or violated)
      - All literals are -2,-1,0,1,2; no hidden constants; all tunables declared in PARAMETER_SCHEMA.
    Smaller score = higher priority."""
    eps = 9.062780215575413e-08
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)

    def robust_norm(x):
        x = np.asarray(x, dtype=float)
        denom = np.mean(np.abs(x)) + eps
        return x / (denom + eps)
    slack_abs = np.abs(slack)
    slack_score = np.where(slack < 0, slack_abs ** 2.054819751383592, slack * 0.37248870664343015)
    rank_median = np.median(upward_rank) if N > 1 else np.mean(upward_rank)
    is_high_rank = upward_rank >= rank_median
    is_tight_or_violated = slack <= 0
    critical_gate = np.where(is_high_rank & is_tight_or_violated, 4.1596492433170535, 1.0)
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = robust_norm(energy_per_sec)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 1.57957962031453
    wait_clipped = np.clip(ready_wait_time, 0, 2.214184726521691)
    wait_score = -robust_norm(wait_clipped)
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 1.601628006938996
    slack_p90 = np.percentile(slack, 88.32319958131757) if N > 1 else np.max(slack)
    slack_p10 = np.percentile(slack, 9.17138065692445) if N > 1 else np.min(slack)
    slack_range = np.maximum(eps, slack_p90 - slack_p10)
    slack_normalized = np.clip((slack - slack_p10) / (slack_range + eps), 0, 1)
    weight_rank = 0.692436046739073 + (1 - 0.692436046739073) * (1 - slack_normalized)
    rank_score = -robust_norm(upward_rank) * weight_rank
    energy_norm = robust_norm(min_incremental_energy)
    unc_norm = robust_norm(uncertainty)
    energy_uncertainty_score = 0.21820237862104608 * energy_norm * unc_norm
    score = robust_norm(slack_score) + robust_norm(unc_slack_coupling) + robust_norm(duration_risk) + 0.9277778169833516 * energy_eff_score + rank_score + wait_score + energy_uncertainty_score + robust_norm(min_incremental_energy) * (1.0 - weight_rank)
    score = score * critical_gate
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score
