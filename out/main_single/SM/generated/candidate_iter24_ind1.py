import numpy as np
RULE_METADATA = {'structure_hash': 'd0557abedfe9d6775cd3278240be43cf2ec975ddfcaae060074359909c1b2f4f', 'parameter_schema_hash': 'df50cdc00a4e04ece98af92664f0b403cd9ed729302c2c1efca1478dbaa11dd5', 'best_parameter_hash': 'bdb837644eafaa230de4b1d887771020087aa689b4386d650d63b337446be209', 'best_parameters': {'epsilon': 0.0006048588437858871, 'slack_risk_penalty': 8.81430641738444, 'slack_urgency_gain': 2.4819147569108253, 'energy_efficiency_weight': 2.403689203412631, 'criticality_weight': 2.068917613226382, 'bottleneck_proximity_weight': 3.8131147021805396, 'wait_ramp_threshold': 21.356138322680934, 'ddl_protection_gate': 0.2307202669840624, 'finfo_max_scale': 138773957.54894787, 'robust_normalization_quantile': 0.6181837638843664, 'successor_release_sharpness': 0.5259915159345231, 'host_load_gating_strength': 0.6225016113135872}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': 'e0fb65074d6a602c92468ae9e5247281cdb6c584aa5d085b357dab5dc6541409', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with key structural improvement:
      - Replaces redundant energy-slack coupling with host-load-aware gating:
        gates energy term using *joint* signal of ready_wait_time (congestion proxy) and uncertainty (risk proxy),
        enabling suppression of marginal energy when system is both busy and uncertain — directly addressing 'avoidable marginal energy' failures.
      - Removes multiplicative uncertainty amplification (previously causing outlier explosion), replaces with bounded additive penalty.
      - Retains quantile normalization, successor-release interaction, and DDL-first slack penalty dominance.
      - All literals are in {-2,-1,0,1,2}; no hidden constants; fully deterministic and finite.
    """
    eps = 0.0006048588437858871
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
            scale = np.quantile(abs_x[finite_mask], 0.6181837638843664)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(0.2307202669840624 * slack_norm))
    slack_penalty = np.where(slk < 0, 8.81430641738444 * np.abs(slack_norm), -2.4819147569108253 * np.abs(slack_norm))
    wait_sig = 1.0 / (1.0 + np.exp(-wait))
    uncert_sig = 1.0 / (1.0 + np.exp(-uncert))
    host_load_gate = (1.0 - wait_sig) * (1.0 - uncert_sig)
    gated_energy_weight = 2.403689203412631 * (1.0 + 0.6225016113135872 * (1.0 - host_load_gate))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -gated_energy_weight * normalize(inv_energy)
    rank_score = -2.068917613226382 * ddl_gate * normalize(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 0.5259915159345231)
    bottleneck_score = -3.8131147021805396 * normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_score = dur_norm + np.clip(0.6225016113135872 * uncert_norm, 0.0, 2.0)
    wait_clipped = np.clip(wait, 0.0, 21.356138322680934)
    wait_normalized = normalize(wait_clipped + eps)
    wait_ramp = np.clip(wait_normalized, 0.0, 1.0)
    wait_score = -wait_ramp
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 138773957.54894787
    min_safe = -finfo.max / 138773957.54894787
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
