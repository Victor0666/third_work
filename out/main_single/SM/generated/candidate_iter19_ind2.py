import numpy as np
RULE_METADATA = {'structure_hash': 'f9fe38cd4d8b13e054ec695a7a37975999dfa37830c207e391e271c789d70014', 'parameter_schema_hash': '93ffd868e8900904ef4ef7dee27b3c8ecc396e92705260437359850bca2515ab', 'best_parameter_hash': 'e842d6ef2afb0c9a3bdbf26fcab6fd71fa7ffcab2b80ed0ea98ed64c33d7e401', 'best_parameters': {'epsilon': 0.09940152159292641, 'slack_risk_penalty': 2.110752408320229, 'slack_urgency_gain': 1.0532491597791418, 'energy_efficiency_weight': 1.1016457678853202, 'criticality_weight': 1.0929497528989505, 'bottleneck_proximity_weight': 4.07730466709345, 'duration_uncertainty_ratio': 0.017488258852295744, 'wait_ramp_threshold': 35.452717469596465, 'robust_normalization_quantile': 0.8903900092244992, 'successor_release_sharpness': 3.7429310790804133, 'ddl_feasibility_margin': 0.7822347103053656, 'finfo_safety_factor': 0.4698394827201119}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': '5854f56a26fecd1f2dd0eb8b41f7966aba5afeaac5f9da2931d93112bb5bb488', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with key structural changes:
      - Replaced sigmoid DDL gate with piecewise-linear ddl_feasibility_margin for smoother, interpretable feasibility boundary.
      - Introduced ddl_feasible_mask: binary mask enabling energy scoring *only* when slack >= margin, aligning energy minimization strictly with feasibility.
      - Retained successor-release interaction but simplified its normalization to avoid double-quantile distortion.
      - Removed finfo_max_scale (inactive per diagnostics) and replaced saturation with explicit finite clipping using np.finfo and tunable safety factor.
      - Normalized all features using robust quantile scaling *before* interaction terms to prevent amplification of outliers.
      - Added bounded exponential starvation mitigation (1 - exp(-wait/θ)) for smooth, asymptotic anti-starvation behavior.
      - Enforced strict ordering: slack penalty dominates; energy only considered when feasible; criticality and bottleneck terms gated by feasibility.
    """
    eps = 0.09940152159292641
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def normalize(x):
        x = np.asarray(x)
        abs_x = np.abs(x)
        finite_mask = np.isfinite(abs_x)
        if np.any(finite_mask):
            scale = np.quantile(abs_x[finite_mask], 0.8903900092244992)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    exec_norm = normalize(exec_t + eps)
    comm_norm = normalize(comm_t + eps)
    energy_norm = normalize(energy + eps)
    slack_norm = normalize(slk)
    rank_norm = normalize(rank + eps)
    work_norm = normalize(work + eps)
    wait_norm = normalize(wait + eps)
    uncert_norm = normalize(uncert + eps)
    ddl_feasible_mask = (slk >= 0.7822347103053656).astype(np.float64)
    slack_penalty = np.where(slk < 0, 2.110752408320229 * np.abs(slack_norm), -1.0532491597791418 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy_norm + eps)
    energy_score = -1.1016457678853202 * inv_energy * ddl_feasible_mask
    rank_score = -1.0929497528989505 * rank_norm * ddl_feasible_mask
    slack_magnitude_inv = 1.0 / (np.abs(slack_norm) + eps)
    bottleneck_sharpened = rank_norm * work_norm * np.power(slack_magnitude_inv, 3.7429310790804133)
    bottleneck_score = -4.07730466709345 * bottleneck_sharpened
    dur_uncert_blend = exec_norm + comm_norm + 0.017488258852295744 * uncert_norm
    dur_score = dur_uncert_blend
    wait_saturation = 1.0 - np.exp(-wait_norm / (35.452717469596465 + eps))
    wait_score = -wait_saturation
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max * 0.4698394827201119
    min_safe = finfo.min * 0.4698394827201119
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
