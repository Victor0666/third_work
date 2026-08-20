import numpy as np
RULE_METADATA = {'structure_hash': 'a42ce69e9e7ad18a9a901059894c18ce8ca5684b7989852568944880f40a5e9a', 'parameter_schema_hash': '97b94cbdcdbe48d8f9d6cb078a8a2860529969afdb88beb93f2761caa9e44ddd', 'best_parameter_hash': 'b11f2b2613b940ed253150de14032ff59044522731d105eba3319a774a09b4c1', 'best_parameters': {'epsilon': 0.000174046539835477, 'slack_risk_penalty': 10.026329119732072, 'slack_urgency_gain': 5.9877154104599395, 'energy_efficiency_weight': 0.05009306834338247, 'criticality_weight': 1.7627157696545532, 'duration_uncertainty_ratio': 0.5463751556197992, 'wait_decay_rate': 0.21563287662227432, 'uncertainty_slack_interaction': 0.35933253950569094, 'bottleneck_boost_weight': 1.1308241900208174, 'ddl_protection_sigmoid_scale': 2.5240965726100355, 'z_score_clip_sigma': 4.393936048419111, 'finfo_max_scale': 141656.00118434025}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': 'c2d6edde333fcb5a54869eb3fe85a5a447994cd3eadd4389179ff8d3ea6ef7fb', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with soft DDL-protection gating and robust z-score clipping.
    
    Key structural improvements:
      - Replaces hard gating (np.where) with **soft sigmoid-based DDL protection**: 
        criticality and energy terms are smoothly scaled by sigmoid(-slack_norm * scale), 
        enabling graceful degradation under increasing deadline pressure instead of abrupt on/off.
      - Introduces **robust z-score normalization with ±sigma clipping**: computes mean/std only over finite values, 
        then clips outliers beyond ±z_score_clip_sigma * std — more discriminative than median alone, 
        and more stable than raw mean-abs across heterogeneous workloads.
      - Retains bottleneck boost but refines activation: now gated by *both* negative slack *and* non-zero remaining_work, 
        avoiding false bottlenecks from rank-only spikes in low-work leaves.
      - All components remain additive, sign-consistent, and shape-preserving; no loops or hidden state.
    """
    eps = 0.000174046539835477
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
        finite_mask = np.isfinite(x) & (np.abs(x) > eps)
        if not np.any(finite_mask):
            return np.zeros_like(x)
        x_finite = x[finite_mask]
        mu = np.mean(x_finite)
        std = np.std(x_finite) if len(x_finite) > 1 else eps
        z = (x - mu) / (std + eps)
        clip_bound = 4.393936048419111
        z_clipped = np.clip(z, -clip_bound, clip_bound)
        max_abs = np.maximum(np.max(np.abs(z_clipped)), eps)
        return z_clipped / (max_abs + eps)
    slack_norm = normalize(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(2.5240965726100355 * slack_norm))
    slack_penalty = np.where(slk < 0, 10.026329119732072 * np.abs(slack_norm), -5.9877154104599395 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.05009306834338247 * normalize(inv_energy) * ddl_gate
    rank_score = -1.7627157696545532 * normalize(rank) * ddl_gate
    bottleneck_boost = rank * work
    bottleneck_active = np.where((slk < 0) & (work > eps), bottleneck_boost, 0.0)
    bottleneck_score = -1.1308241900208174 * normalize(bottleneck_active + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.5463751556197992 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-0.21563287662227432 * wait)
    wait_score = -normalize(wait_sat + eps)
    unc_slack_interaction = 0.35933253950569094 * uncert_norm * np.where(slack_norm < 0, np.abs(slack_norm), 0.0)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score + unc_slack_interaction
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 141656.00118434025
    min_safe = -finfo.max / 141656.00118434025
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
