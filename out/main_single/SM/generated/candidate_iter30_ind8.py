import numpy as np
RULE_METADATA = {'structure_hash': '7cc02c4e7edc71b7729ffc8a7d660639ac22c289a98c08bf5baae211137da5b8', 'parameter_schema_hash': '47b7f4dbb4f584e387b7d14ea476b789ef0ccf8f7dc148f9613ba82db10110e6', 'best_parameter_hash': 'b9fe2c570de678efc464d4e4dbfb883259e80c8ce775f0888225269efaf2b96c', 'best_parameters': {'epsilon': 4.403374430384221e-05, 'slack_risk_penalty': 3.5658821083277115, 'slack_urgency_gain': 1.4738311011989889, 'energy_efficiency_weight': 0.4723244381550469, 'criticality_weight': 1.4083688143235948, 'bottleneck_proximity_weight': 0.88447450916108, 'duration_uncertainty_ratio': 1.7029383282986559, 'wait_ramp_threshold': 50.27854115432801, 'piecewise_slack_transition': 0.09710761950394046, 'robust_normalization_quantile': 0.7413621537560933, 'finfo_clamp_scale': 476717545.06712455, 'ddl_protection_gate_strength': 2.032617341796268}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': 'e33475b10b8c1d0f05edbc0ed7d64ef2c6fbaed166ef4bd50017eba0aaa2883b', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
      - IRREVOCABLE DDL PREFILTERING: tasks with slack < 0 receive dominant multiplicative penalty boost (no energy/criticality override)
      - REPLACED exponential starvation ramp with CLIPPED LINEAR WAIT TERM: stable, interpretable, and insensitive to exponent tuning
      - HOST-LOAD-AWARE INTERACTION: penalizes tasks whose high-work successors have low slack — computed via robust normalized product
      - All normalizations use quantile 0.85 for stronger outlier suppression
      - No exponentiation, no unbounded functions; only safe linear/piecewise forms and finite clamping
      - All numeric coefficients declared in PARAMETER_SCHEMA; only literals are -2,-1,0,1,2
      - Removed 'successor_pressure_weight' to comply with 12-parameter limit; merged its effect into bottleneck_proximity_weight and criticality_weight logic implicitly via shared features.
    """
    eps = 4.403374430384221e-05
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
        x = np.asarray(x, dtype=np.float64)
        abs_x = np.abs(x)
        finite_mask = np.isfinite(abs_x)
        if np.any(finite_mask):
            scale = np.quantile(abs_x[finite_mask], 0.7413621537560933)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    delta = 0.09710761950394046
    ddl_feasible_mask = np.where(slk >= delta, 1.0, 0.0)
    ddl_urgency = np.clip((slk + delta) / (2 * delta), 0.0, 1.0)
    base_slack_penalty = np.where(slk < 0, 3.5658821083277115 * np.abs(slk), -1.4738311011989889 * slk)
    ddl_protection_boost = np.where(slk < 0, 2.032617341796268, 1.0)
    slack_penalty = base_slack_penalty * ddl_protection_boost
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.4723244381550469 * normalize(inv_energy) * ddl_feasible_mask
    rank_score = -1.4083688143235948 * ddl_urgency * normalize(rank + eps)
    bottleneck_term = rank * work / (np.abs(slk) + eps)
    bottleneck_score = -0.88447450916108 * normalize(bottleneck_term + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 1.7029383282986559 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clip = np.clip(wait / (50.27854115432801 + eps), 0.0, 1.0)
    wait_score = -normalize(wait_clip + eps)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 476717545.06712455
    min_safe = -finfo.max / 476717545.06712455
    score = np.clip(score, min_safe, max_safe)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
