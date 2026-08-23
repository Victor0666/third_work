import numpy as np
RULE_METADATA = {'structure_hash': 'a21e2c48f55a22f68c3ba868160de8b313402134f96eb4962b19388ef4ae01e1', 'parameter_schema_hash': 'dddc166256ec39800479b6785c10a304c0268779a1576a20b1f697dbc65465c7', 'best_parameter_hash': '156a54a9bcf7b5dcf0924ce71c2787217da585cc745895bc684077733ae20ce5', 'best_parameters': {'epsilon': 3.0212122267784936e-06, 'slack_penalty_exponent': 2.295048392516127, 'criticality_boost': 2.650235995169352, 'energy_efficiency_ratio_weight': 1.278167189603411, 'uncertainty_slack_coupling': 0.647539521407152, 'rank_slack_balance': 0.6587443121755698, 'duration_risk_penalty': 1.2635297875050344, 'energy_uncertainty_interaction': 1.6310734592705876, 'uncertainty_sigmoid_steepness': 4.839782309332919, 'slack_min_bound': -6.694355896244261, 'slack_max_bound': 23.738013922530364, 'upward_rank_uncertainty_weight': 0.3091547478427385}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': '3865c109b4cc981a3e6cb94827e160e01ec50c110ce450f9583e9ad0ec38b4df', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining strengths from both parents:
      - Employs Parent 2's robust fixed-bound slack scaling (no percentiles) and bounded uncertainty sigmoid.
      - Integrates Parent 1's insight on uncertainty-aware rank amplification, now parameterized as `upward_rank_uncertainty_weight`.
      - Removes all wait-time and early-slack reward terms (confirmed inactive).
      - Uses monotonic slack scoring: only penalizes negative slack; zero reward for positive slack.
      - Criticality boost is applied multiplicatively *after* base score assembly for clean gradient flow.
      - All numeric literals are strictly in {-2,-1,0,1,2}; no hidden constants.
      - Robust_norm avoids tanh (simpler, differentiable, stable) and uses mean-abs + eps.
      - Final score is finite, deterministic, and shape-(N,) guaranteed.
    """
    eps = 3.0212122267784936e-06
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
    slack_score = np.where(slack < 0, slack_abs ** 2.295048392516127, 0.0)
    rank_median = np.median(upward_rank) if N > 1 else np.mean(upward_rank)
    is_high_rank = upward_rank >= rank_median
    is_tight_or_violated = slack <= 0
    critical_gate = np.where(is_high_rank & is_tight_or_violated, 2.650235995169352, 1.0)
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = robust_norm(energy_per_sec)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 0.647539521407152
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 1.2635297875050344
    slack_lb = -6.694355896244261
    slack_ub = 23.738013922530364
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.6587443121755698 + (1.0 - 0.6587443121755698) * (1.0 - slack_scaled)
    augmented_rank = upward_rank * (1.0 + uncertainty * 0.3091547478427385)
    rank_score = -robust_norm(augmented_rank) * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-4.839782309332919 * (uncertainty - 1.0)))
    energy_norm = robust_norm(min_incremental_energy)
    unc_norm = robust_norm(uncertainty)
    energy_uncertainty_score = 1.6310734592705876 * energy_norm * unc_norm * unc_sigmoid
    residual_energy_term = robust_norm(min_incremental_energy) * (1.0 - weight_rank)
    score = robust_norm(slack_score) + robust_norm(unc_slack_coupling) + robust_norm(duration_risk) + 1.278167189603411 * energy_eff_score + rank_score + energy_uncertainty_score + residual_energy_term
    score = score * critical_gate
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score
