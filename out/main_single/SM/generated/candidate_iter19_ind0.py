import numpy as np
RULE_METADATA = {'structure_hash': '67f8e1ca19806c22a2bc2482d370261c7d98ce1457ac2f9a71598842228938ab', 'parameter_schema_hash': 'd0b2ef62e713616e5d149856981a6af79b738489e5666d70aab890f6d97a5c0b', 'best_parameter_hash': '4ad1effb781dc4960392bf0929b4ac68275d2b8045bb7d7f3e90864273fedb29', 'best_parameters': {'epsilon': 1.1961665017896163e-05, 'slack_risk_penalty': 2.494803157443884, 'slack_urgency_gain': 2.2901477536667825, 'energy_efficiency_weight': 2.499369450543422, 'criticality_weight': 0.8459999131047614, 'bottleneck_proximity_weight': 1.586620475091518, 'duration_uncertainty_ratio': 0.6309448175930344, 'wait_ramp_threshold': 29.994256573031148, 'ddl_feasibility_margin': 0.03310296361209816, 'robust_normalization_quantile': 0.7443869317274212, 'successor_release_sharpness': 1.3040132326761165, 'finfo_max_scale': 901009299.977244}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': '27a32c4de1a52debcbd4acebd3e7b55a8d7cd811056d4bb0c1dab96622a7b49c', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with key structural changes:
      - Replaced sigmoid DDL gate with piecewise-linear feasibility margin (more interpretable, less fragile)
      - Introduced ddl_feasible_mask to conditionally enable energy scoring only when slack > margin
      - Removed unused ddl_protection_gate parameter
      - Added robust clipping for all normalized features to [-2, 2] to prevent rank inversions
      - Kept successor-release interaction but re-normalized via absolute quantile for signed slack terms
      - Starvation term now uses bounded exponential saturation (1 - exp(-wait/theta)) for smoother long-tail response
      - All feature interactions remain bounded, deterministic, and free of infinite loops/divisions
    """
    eps = 1.1961665017896163e-05
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def normalize_signed(x):
        x = np.asarray(x)
        finite_mask = np.isfinite(x)
        if np.any(finite_mask):
            scale = np.quantile(np.abs(x[finite_mask]), 0.7443869317274212)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        normed = x / (scale + eps)
        return np.clip(normed, -2.0, 2.0)

    def normalize_positive(x):
        x = np.asarray(x)
        finite_mask = np.isfinite(x) & (x >= 0)
        if np.any(finite_mask):
            scale = np.quantile(x[finite_mask], 0.7443869317274212)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        normed = x / (scale + eps)
        return np.clip(normed, 0.0, 2.0)
    slack_norm = normalize_signed(slk)
    ddl_feasibility_margin = 0.03310296361209816
    ddl_feasible_mask = np.where(slk > ddl_feasibility_margin, 1.0, np.clip((slk - ddl_feasibility_margin) / (eps + ddl_feasibility_margin), 0.0, 1.0))
    slack_penalty = np.where(slk < 0, 2.494803157443884 * np.abs(slack_norm), -2.2901477536667825 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -2.499369450543422 * normalize_positive(inv_energy) * ddl_feasible_mask
    rank_score = -0.8459999131047614 * normalize_positive(rank + eps) * ddl_feasible_mask
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 1.3040132326761165)
    bottleneck_score = -1.586620475091518 * normalize_positive(bottleneck_sharpened + eps) * ddl_feasible_mask
    duration = exec_t + comm_t
    dur_norm = normalize_positive(duration + eps)
    uncert_norm = normalize_positive(uncert + eps)
    dur_uncert_blend = dur_norm + 0.6309448175930344 * uncert_norm
    dur_score = normalize_positive(dur_uncert_blend)
    wait_saturation = 1.0 - np.exp(-wait / (29.994256573031148 + eps))
    wait_score = -np.clip(wait_saturation, 0.0, 1.0)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 901009299.977244
    min_safe = -finfo.max / 901009299.977244
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
