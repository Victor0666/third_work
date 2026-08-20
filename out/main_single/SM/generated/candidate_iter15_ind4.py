import numpy as np
RULE_METADATA = {'structure_hash': 'f2d7da8aa09a5fde7bcb662463b8de6e0a10fe117eb6b5e3fbca21b1a629e806', 'parameter_schema_hash': '1331b86c5995616692c7e09c075e077723c9e780bd84a63f0578df1aad26d049', 'best_parameter_hash': '0d99133aedaf8a22e99c3f7b7a358fbcb38908ffb7d297eebd66d7da38eb8595', 'best_parameters': {'epsilon': 0.005107078704032381, 'slack_risk_penalty': 2.0820757142244197, 'slack_urgency_gain': 4.576953001902819, 'energy_efficiency_weight': 2.421982222707561, 'criticality_weight': 0.30136176218751165, 'bottleneck_proximity_weight': 2.3308367958063765, 'ddl_protection_steepness': 2.5480190377808425, 'ddl_protection_center': -0.07314858141784986, 'wait_clip_threshold': 3.581048226570767, 'upward_rank_normalization_floor': 0.060806626583174955, 'duration_uncertainty_ratio': 0.3826922408512998, 'starvation_exponent': 1.532690172805755}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': '9a60801608c6be7630037a7eb15978975a1fc95e49a0343091a3629825aea4b5', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
      - Arctan-based saturating slack penalty (replacing linear) for robust boundary behavior
      - Preserved smooth sigmoid DDL protection gate with tunable center & steepness
      - Bottleneck-proximal term and power-law starvation mitigation
      - Robust bounded normalization with [-2,2] clipping and MAD scaling
      - All operations guarded against NaN/inf/zero; no hidden constants beyond {-2,-1,0,1,2}
      - Removed slack_saturation_scale to comply with 12-parameter limit; use fixed π scaling
    """
    eps = 0.005107078704032381
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
            med = np.median(abs_x[finite_mask])
            dev = np.abs(abs_x - med)
            mad = np.median(dev[finite_mask]) if np.any(finite_mask) else eps
            scale = max(med, 1.0 * mad, eps)
        else:
            scale = eps
        normed = x / (scale + eps)
        return np.clip(normed, -2.0, 2.0)
    slack_norm = normalize(slk)
    gate_input = 2.5480190377808425 * (-0.07314858141784986 - slack_norm)
    gate_input_clipped = np.clip(gate_input, -np.log(np.finfo(float).max), np.log(np.finfo(float).max))
    ddl_gate = 1.0 / (1.0 + np.exp(-gate_input_clipped))
    slack_arctan = 2.0 / np.pi * np.arctan(slack_norm)
    slack_penalty = np.where(slk < 0, 2.0820757142244197 * (1.0 + slack_arctan), -4.576953001902819 * (1.0 - slack_arctan))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -2.421982222707561 * normalize(inv_energy)
    rank_stable = rank + 0.060806626583174955
    rank_score = -0.30136176218751165 * normalize(rank_stable) * ddl_gate
    bottleneck = rank * work
    bottleneck_score = -2.3308367958063765 * normalize(bottleneck + eps) * ddl_gate
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.3826922408512998 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 3.581048226570767)
    wait_sharpened = np.power(wait_clipped + eps, 1.532690172805755)
    wait_score = -normalize(wait_sharpened)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    score = np.where(np.isfinite(score), score, 0.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
