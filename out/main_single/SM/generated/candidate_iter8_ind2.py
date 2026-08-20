import numpy as np
RULE_METADATA = {'structure_hash': '36d93ab6f843bbc481b8f91230368434f107d3d8af70b4a569e506c2e890e403', 'parameter_schema_hash': 'd7def5d3970a2f270044d8fbc592dd0a04d5c8284486c94a69f5709e8d61bdad', 'best_parameter_hash': 'e89ad64524f2f2e3fe96977b1e311a577633fccfede0f7da1fc9a5730f92d214', 'best_parameters': {'epsilon': 0.040794855616112825, 'slack_risk_penalty': 9.587572136972922, 'slack_urgency_gain': 4.079601164934894, 'energy_efficiency_weight': 0.5038820698964785, 'criticality_weight': 2.1683291841924426, 'duration_uncertainty_ratio': 0.364813403025804, 'wait_clip_threshold': 62.74641022363573, 'bottleneck_proximity_weight': 0.24180339397267728, 'ddl_protection_gate': 0.23453351445913867, 'finfo_max_scale': 104526.13764738665, 'load_risk_exponent': 1.5032042311966993, 'robust_normalization_factor': 0.6106531639433549}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': 'b3df105450dffc8c181ff62f113f0de1222ce1341b3e180becb89f90bd0ecdec', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's stability with Parent 1's risk-aware interactions:
      - Retains hard wait clipping and bottleneck-proximal term from Parent 2.
      - Integrates robust median/MAD normalization with tunable robustness factor.
      - Adds load-risk interaction (work * uncertainty)^exponent, gated by ddl_gate to prioritize under deadline stress.
      - Uses sigmoid DDL gate for smooth feasibility enforcement across all critical terms.
      - All numeric literals strictly in {-2,-1,0,1,2}; no hidden constants.
      - Deterministic, finite, shape-compliant, and fully parameterized.
    """
    eps = 0.040794855616112825
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
        if not np.any(finite_mask):
            return np.zeros_like(x)
        med = np.median(abs_x[finite_mask])
        mad = np.median(np.abs(abs_x[finite_mask] - med))
        scale = max(med, 0.6106531639433549 * mad, eps)
        return x / (scale + eps)
    slack_norm = normalize(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(0.23453351445913867 * slack_norm))
    slack_penalty = np.where(slk < 0, 9.587572136972922 * np.abs(slack_norm), -4.079601164934894 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.5038820698964785 * normalize(inv_energy)
    rank_score = -2.1683291841924426 * ddl_gate * normalize(rank + eps)
    bottleneck = rank * work
    bottleneck_score = -0.24180339397267728 * normalize(bottleneck + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.364813403025804 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 62.74641022363573)
    wait_score = -normalize(wait_clipped + eps)
    load_risk = (work * uncert + eps) ** 1.5032042311966993
    load_risk_score = normalize(load_risk) * ddl_gate
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score + load_risk_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 104526.13764738665
    min_safe = -finfo.max / 104526.13764738665
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
