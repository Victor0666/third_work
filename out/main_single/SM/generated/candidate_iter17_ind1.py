import numpy as np
RULE_METADATA = {'structure_hash': 'affc04784b6c0824a138afbfd95decaeb40baacf568b59d38004ae3e62b9706a', 'parameter_schema_hash': '52db11f1568990e3673e5458170f978f6785a1a1432f47b95b1c623715e4196b', 'best_parameter_hash': '525ad0106491a6e79f7dd26a6de877e589fa2bf61c72d5982510649bf8421cd6', 'best_parameters': {'epsilon': 0.004357080983022363, 'slack_risk_penalty': 5.1978445409957015, 'slack_urgency_gain': 1.2380423213224345, 'energy_efficiency_weight': 1.0342107447155013, 'criticality_weight': 1.708690844709157, 'bottleneck_proximity_weight': 0.8793084392264506, 'duration_uncertainty_ratio': 0.8417061545682784, 'wait_ramp_threshold': 65.43736737310695, 'ddl_protection_gate': 0.21508284060045557, 'finfo_max_scale': 4557.36088186135, 'robust_normalization_quantile': 0.5017523513412874, 'successor_release_sharpness': 2.5860830638683083}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': '726addd024df6de07150f522d509acabb769aff22741d7bbade394d45ac2cc86', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule with:
      - Retained Parent 2's proven successor-release interaction and finfo_max_scale stability
      - Reintroduced *light* dynamic energy suppression via ddl_gate^exponent (simpler than Parent 1's sensitivity coupling)
      - Removed host_load_sensitivity_exponent (redundant with ddl_gate exponentiation) to stay within 12 parameters
      - Keeps bounded linear wait ramp and quantile-based robust normalization
      - Slack penalty remains dominant to enforce hard deadline feasibility first
      - All numeric literals are in {-2,-1,0,1,2}; eps and bounds handled via PARAMS or np.finfo
    """
    eps = 0.004357080983022363
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
            scale = np.quantile(abs_x[finite_mask], 0.5017523513412874)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(0.21508284060045557 * slack_norm))
    slack_penalty = np.where(slk < 0, 5.1978445409957015 * np.abs(slack_norm), -1.2380423213224345 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.0342107447155013 * normalize(inv_energy) * np.power(ddl_gate, 1.0)
    rank_score = -1.708690844709157 * ddl_gate * normalize(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 2.5860830638683083)
    bottleneck_score = -0.8793084392264506 * normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.8417061545682784 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 65.43736737310695)
    wait_normalized = normalize(wait_clipped + eps)
    wait_ramp = np.clip(wait_normalized, 0.0, 1.0)
    wait_score = -wait_ramp
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 4557.36088186135
    min_safe = -finfo.max / 4557.36088186135
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
