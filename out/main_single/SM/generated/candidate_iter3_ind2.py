import numpy as np
RULE_METADATA = {'structure_hash': 'f8cdf8d32cea969da8d4fe8b7dc970d0b33d235fd37c445203625548d5df1e57', 'parameter_schema_hash': 'a6d38449188891b680b94bc79494786e1a637ad35caafb4198e823571c9a55e7', 'best_parameter_hash': '87581e89e993ff68ed3d0e515455ff88b3b518fc0bd5ed504ff63f534aba7988', 'best_parameters': {'epsilon': 0.00012284828451808983, 'slack_risk_penalty': 0.8313034268904758, 'slack_urgency_gain': 2.898362817271544, 'energy_efficiency_weight': 3.1894112559272223, 'criticality_weight': 1.8193008950100844, 'duration_uncertainty_ratio': 1.859388966655728, 'wait_decay_rate': 0.05387746497639204, 'percentile_range_scale': 0.8407896165925669, 'pctl_low': 10.576039082305766, 'pctl_high': 82.51939392841146}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': '2d3874c4465d65654aa7305f9ad7b3285ee6f3e60e2992bd523779a71e9ac8c5', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with strict DDL-protection gating, percentile-based normalization,
    and eliminated fragile uncertainty-slack interaction.
    
    Key structural improvements:
      - Replaced MAD-based normalization with bounded configurable percentile scaling (pctl_low/pctl_high).
      - Hard DDL-protection gate: zero out *both* upward_rank and energy_score contributions unless
        slack >= 0 — enforces strict feasibility-first ordering before any energy optimization.
      - Removed uncertainty_slack_interaction entirely — confirmed inactive and adds numerical fragility.
      - All components now use the same robust percentile-based normalizer for consistency and stability.
    """
    eps = 0.00012284828451808983
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def percentile_normalize(x):
        x = np.asarray(x)
        p_low = np.percentile(x, 10.576039082305766)
        p_high = np.percentile(x, 82.51939392841146)
        rng = p_high - p_low
        rng_safe = np.where(np.isfinite(rng) & (rng > eps), rng, eps)
        center = (p_low + p_high) / 2.0
        return (x - center) / (0.8407896165925669 * rng_safe + eps)
    duration = exec_t + comm_t
    slack_norm = percentile_normalize(slk)
    slack_penalty = np.where(slk < 0, 0.8313034268904758 * slack_norm ** 2, -2.898362817271544 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_active = np.where(slk >= 0, inv_energy, 0.0)
    energy_score = -3.1894112559272223 * percentile_normalize(energy_active + eps)
    rank_active = np.where(slk >= 0, rank, 0.0)
    rank_score = -1.8193008950100844 * percentile_normalize(rank_active + eps)
    dur_norm = percentile_normalize(duration + eps)
    uncert_norm = percentile_normalize(uncert + eps)
    dur_uncert_blend = (1.0 + 1.859388966655728 * uncert_norm) / (dur_norm + eps + 1.859388966655728 * uncert_norm + eps)
    dur_score = percentile_normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-0.05387746497639204 * wait)
    wait_score = -percentile_normalize(wait_sat + eps)
    score = slack_penalty + energy_score + rank_score + dur_score + wait_score
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
