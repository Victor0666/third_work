import numpy as np
RULE_METADATA = {'structure_hash': 'f6703a61b6b29a53b652025fd24d83ce36947390e6b68bb5c8621c997c26eba8', 'parameter_schema_hash': 'f36e1ad3ae55ea62070d3332f5abe3c16bfcaa55a03372049b1ae61104c9afe6', 'best_parameter_hash': 'f15eff25f9a1cef5047ea71fe61be6f7db129a7ba4734d889067400e1570156c', 'best_parameters': {'epsilon': 0.07912650648483613, 'slack_risk_penalty': 9.44913152155499, 'slack_urgency_gain': 2.5657149817321505, 'energy_efficiency_weight': 1.5120039741093028, 'criticality_weight': 1.711740364735185, 'bottleneck_proximity_weight': 2.3131756088297055, 'duration_uncertainty_ratio': 0.47223474200164095, 'wait_ramp_threshold': 1.0759824302132326, 'ddl_protection_gate': 0.21995583057606483, 'robust_normalization_quantile': 0.896621045942271, 'successor_release_sharpness': 2.070420115270074, 'energy_modulation_exponent': 0.6336728193364182}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': '8128a52195c78d2e5ffc6ec0da4380b12f6bf286f71b150c9bcb26903e66e6e7', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining strengths from both parents:
      - Keeps Parent 2's robust sigmoid DDL protection gate (smoother than piecewise-linear).
      - Adopts Parent 2's successor-release interaction with sharpened bottleneck term.
      - Introduces novel *energy modulation exponent*: softens energy term via ddl_gate^exponent instead of hard gating or linear scaling,
        preserving energy-awareness under mild pressure while suppressing it aggressively when slack approaches zero.
      - Removes redundant clipping bounds (score_clipping_bound, finfo_max_scale) — relies on quantile normalization + ε-safeguards + nan_to_num.
      - Uses unified robust normalization with adaptive quantile scaling across all features.
      - Maintains strict DDL-first semantics via dominant slack_penalty and ddl_gate modulation.
      - All operations are finite, deterministic, shape-preserving, and use only allowed literals (-2,-1,0,1,2).
    """
    eps = 0.07912650648483613
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
            scale = np.quantile(abs_x[finite_mask], 0.896621045942271)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(0.21995583057606483 * slack_norm))
    slack_penalty = np.where(slk < 0, 9.44913152155499 * np.abs(slack_norm), -2.5657149817321505 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.5120039741093028 * normalize(inv_energy) * np.power(ddl_gate, 0.6336728193364182)
    rank_score = -1.711740364735185 * ddl_gate * normalize(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 2.070420115270074)
    bottleneck_score = -2.3131756088297055 * normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.47223474200164095 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 1.0759824302132326)
    wait_normalized = normalize(wait_clipped + eps)
    wait_ramp = np.clip(wait_normalized, 0.0, 1.0)
    wait_score = -wait_ramp
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
