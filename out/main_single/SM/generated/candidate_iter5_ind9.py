import numpy as np
RULE_METADATA = {'structure_hash': '4c4fc3e07c71e855fdb8945763ddb58b7bc0454e2c94b7351f5e82810d666908', 'parameter_schema_hash': 'ec9bc549474bae885aeba6f814e8a0c2987755edc7e5acf48c029d5f73c33284', 'best_parameter_hash': '5e3ef6aa39441ad5e03bdaab01ad3926ab73a1349c3691cc17abab91aacf6630', 'best_parameters': {'epsilon': 0.0011564097968489344, 'slack_risk_penalty': 8.63564326590441, 'slack_urgency_gain': 3.360605799936033, 'energy_efficiency_weight': 2.9486160642097157, 'criticality_weight': 0.9793686221661562, 'duration_uncertainty_ratio': 1.0897232323572978, 'wait_decay_rate': 0.1354985871049934, 'uncertainty_slack_interaction': 2.611048368642992, 'finfo_max_scale': 556340.9950031742, 'starvation_sensitivity_exponent': 1.0326931024611934, 'energy_slack_coupling_strength': 1.0544745351843978, 'energy_slack_activation_threshold': 0.18324907383521322}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': '07cb989bc7993becc43b2c00aea3a00ec985416c79841f8139aeea391e462481', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's stability with novel slack-energy coupling and convex starvation response.
    
    Key improvements:
      - Keeps Parent 2's robust normalized-slack gating and bounded linear duration-uncertainty blend.
      - Adds *energy-slack coupling*: boosts priority of low-energy tasks only when slack is sufficiently positive,
        improving energy-awareness without violating DDL-first semantics.
      - Replaces linear wait saturation with *convex exponential saturation* (wait^exponent) for sharper starvation response.
      - Uses unified robust normalization with median/MAD fallback when mean fails (more outlier-resilient than Parent 2).
      - All numeric literals restricted to {-2,-1,0,1,2}; all parameters accessed via PARAMS.
    """
    eps = 0.0011564097968489344
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def robust_normalize(x):
        x = np.asarray(x)
        finite_mask = np.isfinite(x)
        if not np.any(finite_mask):
            return np.zeros_like(x)
        x_clean = x[finite_mask]
        center = np.median(x_clean)
        mad = np.median(np.abs(x_clean - center)) + eps
        scale = max(np.mean(np.abs(x_clean - center)) + eps, mad)
        return (x - center) / (scale + eps)
    slack_norm = robust_normalize(slk)
    slack_penalty = np.where(slk < 0, 8.63564326590441 * np.abs(slack_norm), -3.360605799936033 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -2.9486160642097157 * robust_normalize(inv_energy)
    rank_active = np.where(slack_norm >= 0, rank, 0.0)
    rank_score = -0.9793686221661562 * robust_normalize(rank_active + eps)
    duration = exec_t + comm_t
    dur_norm = robust_normalize(duration + eps)
    uncert_norm = robust_normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 1.0897232323572978 * uncert_norm
    dur_score = robust_normalize(dur_uncert_blend)
    wait_power = np.power(np.maximum(wait, 0.0), 1.0326931024611934)
    wait_sat = 1.0 - np.exp(-0.1354985871049934 * wait_power)
    wait_score = -robust_normalize(wait_sat + eps)
    unc_slack_interaction = 2.611048368642992 * uncert_norm * np.where(slack_norm < 0, np.abs(slack_norm), 0.0)
    energy_slack_boost = np.where(slack_norm > 0.18324907383521322, 1.0544745351843978 * robust_normalize(inv_energy), 0.0)
    score = slack_penalty + energy_score + rank_score + dur_score + wait_score + unc_slack_interaction + energy_slack_boost
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 556340.9950031742
    min_safe = -finfo.max / 556340.9950031742
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
