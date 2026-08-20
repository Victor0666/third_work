import numpy as np
RULE_METADATA = {'structure_hash': '5a658ebf39bc31ea919d375f2c66b11a166477efb115f106bc90909c1fec00c6', 'parameter_schema_hash': '173cd54032920af60948c8c49a489eadc2c707d38554312deacb08401f88a452', 'best_parameter_hash': '7775aae0479d5bb02181358c0741322c435a932dcbe489855d0d59ae99a16866', 'best_parameters': {'epsilon': 0.0077174331048665305, 'slack_risk_penalty': 0.5013922633549001, 'slack_urgency_gain': 2.829632911881727, 'energy_efficiency_weight': 1.427446276555658, 'criticality_weight': 1.7801634094534364, 'bottleneck_proximity_weight': 2.5776981698604566, 'successor_release_sharpness': 2.0474047062617524, 'wait_ramp_threshold': 44.61146442641612, 'ddl_protection_gate': 0.07772574520881066, 'robust_normalization_quantile': 0.8683968705314778, 'duration_uncertainty_ratio': 0.4962916092845367, 'finfo_max_scale': 1066387.969492827}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': '44e87a20b2d106c416dd645408c6e5eb952af67d4c10b655e0086ee7829cb2b0', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with three key structural improvements:
      - Replaces multiplicative uncertainty gating with additive, bounded linear blend (exec_t + comm_t + ratio * uncert)
        to prevent over-suppression and preserve physical interpretability — directly addressing reflection.
      - Removes dual gating on bottleneck term, retaining only the sigmoid ddl_gate for clarity and stability.
      - Introduces *risk-aware slack scaling*: slack penalty is now scaled by (1 + uncertainty) to increase urgency
        proportionally to risk when slack is negative, without affecting positive-slack gain — improves DDL adherence
        under high uncertainty while preserving fairness.
      - All normalizations remain adaptive quantile-based; no hidden constants or unbounded operations.
      - Strict shape-(N,) output with finite values enforced via np.nan_to_num and finfo safeguards.
    """
    eps = 0.0077174331048665305
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
            scale = np.quantile(abs_x[finite_mask], 0.8683968705314778)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(0.07772574520881066 * slack_norm))
    risk_amplified_penalty = np.where(slk < 0, 0.5013922633549001 * np.abs(slack_norm) * (1.0 + np.clip(uncert, 0.0, 2.0)), 0.0)
    slack_gain = np.where(slk > 0, -2.829632911881727 * np.abs(slack_norm), 0.0)
    slack_penalty = risk_amplified_penalty + slack_gain
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.427446276555658 * normalize(inv_energy)
    rank_score = -1.7801634094534364 * ddl_gate * normalize(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    sharpness_factor = np.where(slk > 0, np.power(slack_magnitude_inv, 2.0474047062617524), 1.0)
    bottleneck_sharpened = rank * work * sharpness_factor
    bottleneck_score = -2.5776981698604566 * ddl_gate * normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_uncert_blend = duration + 0.4962916092845367 * np.clip(uncert, 0.0, 2.0)
    dur_score = normalize(dur_uncert_blend + eps)
    wait_clipped = np.clip(wait, 0.0, 44.61146442641612)
    wait_normalized = normalize(wait_clipped + eps)
    wait_ramp = np.clip(wait_normalized, 0.0, 1.0)
    wait_score = -wait_ramp
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 1066387.969492827
    min_safe = -finfo.max / 1066387.969492827
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
