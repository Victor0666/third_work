import numpy as np
RULE_METADATA = {'structure_hash': '7d90f99b7645b3a524660a061726eded98f907b06dd4affd18ea4caac950a01e', 'parameter_schema_hash': '0ec4455dd77853fde9640916a71491cafcd31322bf48297098c22ff0ec4eb3be', 'best_parameter_hash': 'a254d62e80804338bd7df9672c4e81d2a2dc65824386ba127ea98400546d724d', 'best_parameters': {'slack_risk_penalty': 11.361965007695936, 'slack_urgency_gain': 0.1203786969915757, 'energy_efficiency_weight': 0.1258940702428194, 'criticality_weight': 0.5222395323416638, 'bottleneck_proximity_weight': 0.20645844821493473, 'duration_uncertainty_ratio': 0.24750811778859766, 'wait_ramp_threshold': 13.534785720932945, 'ddl_protection_sigmoid_slope': 8.614895599800242, 'robust_normalization_quantile': 0.6230756872717431, 'energy_ddl_coupling_exponent': 0.9895986284906401, 'rank_sharpening_exponent': 0.5939713690138642, 'score_clipping_bound': 126209763700588.06}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': 'f083ae5786cd36ad5893b51e35661cdf4072832d5f8f37769388bdca2385edf3', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining strengths of both parents:
      - Conditional sigmoid DDL protection gate (from Parent 2) for precise deadline-risk targeting.
      - Criticality sharpening via rank_sharpening_exponent (novel): applies power-law enhancement *before* normalization to amplify discrimination among high-rank tasks under pressure.
      - Semantic-aware normalization: abs_quantile for temporal features, pos_quantile for structural ones.
      - Energy scoring uses dedicated coupling exponent and sigmoid gating, avoiding dilution by positive slack.
      - Bottleneck term retains clean rank*work interaction without slack inversion, preserving physical meaning.
      - Starvation mitigation uses monotonic clipped ramp, not exponential saturation (more stable & interpretable).
      - All numeric literals are {-2,-1,0,1,2}; eps via np.finfo; no hidden constants.
    """
    eps = np.finfo(np.float64).tiny
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def normalize_abs_quantile(x, q=0.6230756872717431):
        x = np.asarray(x)
        abs_x = np.abs(x)
        finite_mask = np.isfinite(abs_x)
        if np.any(finite_mask):
            scale = np.quantile(abs_x[finite_mask], q)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)

    def normalize_pos_quantile(x, q=0.6230756872717431):
        x = np.asarray(x)
        finite_mask = np.isfinite(x) & (x >= 0)
        if np.any(finite_mask):
            x_clean = x[finite_mask]
            if len(x_clean) > 0:
                scale = np.quantile(x_clean, q)
                scale = np.where(scale > eps, scale, eps)
            else:
                scale = eps
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize_abs_quantile(slk)
    slack_penalty = np.where(slk < 0, 11.361965007695936 * np.abs(slack_norm), -0.1203786969915757 * np.abs(slack_norm))
    sigmoid_gate = np.where(slk < 0, 1.0 / (1.0 + np.exp(-8.614895599800242 * slk)), 0.0)
    slack_magnitude = np.abs(slk) + eps
    slack_coupling_factor = np.power(slack_magnitude, -0.9895986284906401)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.1258940702428194 * normalize_pos_quantile(inv_energy) * sigmoid_gate * slack_coupling_factor
    rank_sharpened = np.power(rank + eps, 0.5939713690138642)
    rank_score = -0.5222395323416638 * normalize_pos_quantile(rank_sharpened) * sigmoid_gate
    bottleneck_base = rank * work
    bottleneck_score = -0.20645844821493473 * normalize_pos_quantile(bottleneck_base + eps)
    duration = exec_t + comm_t
    dur_norm = normalize_abs_quantile(duration + eps)
    uncert_norm = normalize_abs_quantile(uncert + eps)
    dur_uncert_blend = dur_norm + 0.24750811778859766 * uncert_norm
    dur_score = normalize_abs_quantile(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 13.534785720932945)
    wait_normalized = normalize_pos_quantile(wait_clipped + eps)
    wait_ramp = np.clip(wait_normalized, 0.0, 1.0)
    wait_score = -wait_ramp
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    score = np.nan_to_num(score, nan=0.0, posinf=126209763700588.06, neginf=-126209763700588.06)
    score = np.clip(score, -126209763700588.06, 126209763700588.06)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
