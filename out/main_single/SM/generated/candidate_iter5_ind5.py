import numpy as np
RULE_METADATA = {'structure_hash': '78f1978ac40369f0985141b40365f194210fb2938912b77dc1ec64bff779ae8d', 'parameter_schema_hash': 'b19f155762ac5a2a0ecbf88178b01f30bf8dfc3d9870febd1e908a1e57330542', 'best_parameter_hash': 'a42350d86d0bf0ceb66f34c653406500cc68baa794bb222fb0c5a71c24a1d9ff', 'best_parameters': {'epsilon': 4.5506679513941115e-05, 'slack_risk_penalty': 12.033079152077583, 'slack_urgency_gain': 5.534058142767418, 'energy_efficiency_weight': 1.4898863235899442, 'criticality_weight': 0.5048608335448477, 'duration_uncertainty_ratio': 0.6379518533316768, 'wait_decay_rate': 0.21835896644537903, 'pctl_low': 6.817803439550327, 'pctl_high': 96.33544189284797, 'successor_release_weight': 1.8909118508530396, 'finfo_max_scale': 71932.55576894282}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': '97dd4f51827393a88bda7f62b90aaa1dfef2a7c4cbaf76b6f44d20c7605f1c72', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's strict DDL gating and percentile normalization
    with Parent 1's successor-release interaction — now *conditionally activated only under slack<0*.
    
    Structural improvements:
      - Strict hard gate on all non-slack components: energy_score and rank_score only active when slack >= 0.
      - Successor-release term added *only when slack < 0*, acting as a corrective release mechanism for high-risk paths.
      - Wider percentile range (5th–95th) improves robustness across heterogeneous workflow scales.
      - All normalizations use same percentile_normalize() for consistency and stability.
      - Removed all fragile uncertainty-slack couplings and host-load modulations per performance analysis.
      - Uses quadratic slack penalty (Parent 2) + linear urgency gain (Parent 2), but with improved clipping.
    """
    eps = 4.5506679513941115e-05
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
        p_low = np.percentile(x, 6.817803439550327)
        p_high = np.percentile(x, 96.33544189284797)
        rng = p_high - p_low
        rng_safe = np.where(np.isfinite(rng) & (rng > eps), rng, eps)
        center = (p_low + p_high) / 2.0
        return (x - center) / (rng_safe + eps)
    duration = exec_t + comm_t
    slack_norm = percentile_normalize(slk)
    slack_penalty = np.where(slk < 0, 12.033079152077583 * np.abs(slack_norm) ** 2, -5.534058142767418 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_active = np.where(slk >= 0, inv_energy, 0.0)
    energy_score = -1.4898863235899442 * percentile_normalize(energy_active + eps)
    rank_active = np.where(slk >= 0, rank, 0.0)
    rank_score = -0.5048608335448477 * percentile_normalize(rank_active + eps)
    successor_active = np.where(slk < 0, work, 0.0)
    successor_norm = percentile_normalize(successor_active + eps)
    urgency_norm = percentile_normalize(np.clip(-slk, 0.0, np.inf) + eps)
    successor_score = -1.8909118508530396 * successor_norm * urgency_norm
    dur_norm = percentile_normalize(duration + eps)
    uncert_norm = percentile_normalize(uncert + eps)
    denom = dur_norm + 0.6379518533316768 * uncert_norm + eps
    dur_uncert_blend = (1.0 + 0.6379518533316768 * uncert_norm) / denom
    dur_score = percentile_normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-0.21835896644537903 * wait)
    wait_score = -percentile_normalize(wait_sat + eps)
    score = slack_penalty + energy_score + rank_score + successor_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 71932.55576894282
    min_safe = -finfo.max / 71932.55576894282
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
