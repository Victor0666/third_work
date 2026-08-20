import numpy as np
RULE_METADATA = {'structure_hash': 'b147c7e7301403b6c4946106d47634f024a74733dfab1f52df1608e3851de1aa', 'parameter_schema_hash': '50e96db0ab26cfb24124d45a7cf6dfb7ee3ae82a228745347d1038856bc9105f', 'best_parameter_hash': 'c5fb473a2da7ef8fad64d8ac9373bbb19a6adf15efe2c85fe2b84fa12a2c306f', 'best_parameters': {'epsilon': 0.0018526089738866343, 'slack_risk_penalty': 5.940603545481795, 'slack_urgency_gain': 3.7386278698059914, 'energy_efficiency_weight': 1.6112795975879834, 'criticality_weight': 0.003531008426344422, 'bottleneck_proximity_weight': 1.5210231526725442, 'duration_uncertainty_ratio': 0.8437906282035971, 'wait_ramp_threshold': 18.361099488230252, 'ddl_protection_gate': 0.050543921294071856, 'finfo_max_scale': 13928977.132259315, 'robust_minmax_quantile': 0.861077612575191, 'starvation_slack_threshold': 1.7290549516415727}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': 'd393d0e3df9d0c9ec7887417e5b470176fa10ba0b6a770453c61e96e09d5811d', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with two key structural improvements:
      - Robust min-max normalization per feature using high quantile for range estimation (replaces quantile scaling).
      - Critical-path starvation guard: when slack < starvation_slack_threshold, prioritize bottleneck_proximity unconditionally.
      - Energy suppression simplified to piecewise-linear decay from full weight at slack=0+ to zero at slack=0 — implemented via max(0, 1 - k*|slk|) but bounded.
      - All numeric literals are in {-2,-1,0,1,2}; no hidden constants.
      - Removed 'energy_suppression_slope' to comply with 12-parameter limit; reuse ddl_protection_gate for slope proxy if needed, but here use fixed logic.
    """
    eps = 0.0018526089738866343
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
        finite_mask = np.isfinite(x)
        if np.any(finite_mask):
            x_finite = x[finite_mask]
            q_low = np.quantile(x_finite, 1.0 - 0.861077612575191)
            q_high = np.quantile(x_finite, 0.861077612575191)
            scale = np.where(q_high > q_low, q_high - q_low, eps)
            center = np.median(x_finite)
            return (x - center) / (scale + eps)
        else:
            return np.zeros_like(x)
    slack_penalty = np.where(slk < 0, 5.940603545481795 * np.abs(slk), -3.7386278698059914 * slk)
    energy_suppression_weight = np.where(slk < 0, 1.0, np.clip(1.0 - slk / (0.0018526089738866343 + eps), 0.0, 1.0))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.6112795975879834 * energy_suppression_weight * normalize(inv_energy)
    ddl_gate = 1.0 / (1.0 + np.exp(0.050543921294071856 * slk))
    rank_score = -0.003531008426344422 * ddl_gate * normalize(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 2)
    bottleneck_score = -1.5210231526725442 * normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.8437906282035971 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 18.361099488230252)
    wait_normalized = normalize(wait_clipped + eps)
    wait_ramp = np.clip(wait_normalized, 0.0, 1.0)
    wait_score = -wait_ramp
    starvation_mask = (slk < 1.7290549516415727).astype(np.float64)
    starvation_bypass = bottleneck_score * starvation_mask
    base_score = slack_penalty + energy_score + rank_score + dur_score + wait_score
    score = np.where(starvation_mask > 0, starvation_bypass, base_score)
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 13928977.132259315
    min_safe = -finfo.max / 13928977.132259315
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
