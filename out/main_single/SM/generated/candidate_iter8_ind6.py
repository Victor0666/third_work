import numpy as np
RULE_METADATA = {'structure_hash': '100b0a1be2c6d4524b2f0a2cf672da52614ce555497696755aaa87bee28ff5ba', 'parameter_schema_hash': '304bef76f1fa0048f7ae86d259def632561d4a3303548658404423eab3c39a3d', 'best_parameter_hash': '0c5f0c282b83d71c1e0f5ac4c4fc8ed96d6a92b4ee905cf98e4083b2d4b0a56d', 'best_parameters': {'epsilon': 2.8116094137193883e-05, 'slack_risk_penalty': 7.6602725099828435, 'slack_urgency_gain': 1.4085550676758998, 'energy_efficiency_weight': 1.2180814326539409, 'criticality_weight': 1.407721891686671, 'duration_uncertainty_ratio': 0.25387439122212974, 'wait_clip_threshold': 4.39410738970715, 'bottleneck_proximity_weight': 1.6129476445610234, 'ddl_protection_gate': 0.19183871302289218, 'finfo_max_scale': 61779004.08151271, 'load_uncertainty_exponent': 1.0570456812644542, 'normalized_slack_gate_width': 0.0841902881761587}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': '0adc9e3eb351c9288bd3b20abd64521dc9e1527b3f477b44dc1ecd78617a9327', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's robustness with Parent 1's risk-aware load coupling:
      - Retains hard wait clipping & bottleneck term from Parent 2 (proven high performance)
      - Reintroduces gated nonlinear load-uncertainty interaction (from Parent 1) but now gated by ddl_gate for safety
      - Uses median-based normalization universally (more outlier-resilient than mean-abs)
      - Sharpens DDL gate via normalized_slack_gate_width for tighter feasibility control
      - All numeric literals strictly in {-2,-1,0,1,2}; no hidden constants
      - Deterministic, finite, shape-compliant, fully parameterized
    """
    eps = 2.8116094137193883e-05
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
            med = np.median(abs_x[finite_mask])
            mad = np.median(np.abs(abs_x[finite_mask] - med))
            scale = np.maximum(med, mad, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    gate_arg = 0.0841902881761587 * slack_norm
    ddl_gate = 1.0 / (1.0 + np.exp(0.19183871302289218 * gate_arg))
    slack_penalty = np.where(slk < 0, 7.6602725099828435, -1.4085550676758998) * np.abs(slack_norm)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.2180814326539409 * normalize(inv_energy)
    rank_score = -1.407721891686671 * ddl_gate * normalize(rank + eps)
    bottleneck = rank * work
    bottleneck_score = -1.6129476445610234 * normalize(bottleneck + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.25387439122212974 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 4.39410738970715)
    wait_score = -normalize(wait_clipped + eps)
    load_factor = (work * uncert + eps) ** 1.0570456812644542
    load_score = normalize(load_factor) * ddl_gate
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score + load_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 61779004.08151271
    min_safe = -finfo.max / 61779004.08151271
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
