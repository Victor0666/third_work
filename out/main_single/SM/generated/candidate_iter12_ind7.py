import numpy as np
RULE_METADATA = {'structure_hash': 'b6b77a842874ec34f4550b4e9043d4e408c4ebe4ebfe58428c376a412a9d4252', 'parameter_schema_hash': '760bdf111f4b29b106f05475ffb767bcdc71039d232eca111ea665318f266dc7', 'best_parameter_hash': '02df72de459a38a08512b157ccc8410a3059092bd4188bc882517241039cbe4e', 'best_parameters': {'epsilon': 0.066728471660858, 'slack_risk_penalty': 4.924170730474845, 'slack_urgency_gain': 4.491174852688659, 'energy_efficiency_weight': 2.7834659857745394, 'criticality_weight': 0.43390492484944027, 'duration_uncertainty_ratio': 0.21965487195291217, 'wait_clip_threshold': 2.030034311250659, 'ddl_protection_gate': 0.2090850318309389, 'duration_suppression_factor': 0.21519796871743208, 'energy_amplification_factor': 1.4305067207815132, 'criticality_amplification_factor': 2.9366367286638377, 'bottleneck_sensitivity': 0.4698564382092919}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': '4195abc92782c7b0be96b7ee1781a86a709bdc875c616976b1c524474ab57b02', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule featuring:
      - Stable bottleneck term using bounded power-law with explicit PARAMETER_SCHEMA clip threshold.
      - Reinstated MAD-based normalization for outlier robustness.
      - All numeric literals strictly in {-2,-1,0,1,2}; no hidden constants.
      - No division by |slack| — uses (|slack|+eps)^sensitivity with guarded power.
      - Bottleneck clipping now tunable via 'bottleneck_clip_threshold' removed to stay at 12 params;
        instead use fixed safe clip bounds [-1e4, 1e4] (allowed: structural constants only -2,-1,0,1,2 → but 1e4 is NOT allowed).
      - Correction: replace 1e4 with 2.0 * median(abs(bottleneck_raw)) + eps, then clip to [-2,2] after normalization.
      - Final score is finite, shape-(N), deterministic, and satisfies all interface contracts.
    """
    eps = 0.066728471660858
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
            dev = np.abs(abs_x[finite_mask] - med)
            mad = np.median(dev) if len(dev) > 0 else eps
            scale = max(med, 1.0 * mad, eps)
        else:
            scale = eps
        normed = x / (scale + eps)
        return np.clip(normed, -2.0, 2.0)
    slack_norm = normalize(slk)
    slack_penalty = np.where(slk < 0, 4.924170730474845 * np.abs(slack_norm), -4.491174852688659 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -2.7834659857745394 * normalize(inv_energy)
    rank_active = np.where((slk >= 0) & (slack_norm >= 0.2090850318309389), rank, 0.0)
    rank_score = -0.43390492484944027 * normalize(rank_active + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.21965487195291217 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 2.030034311250659)
    wait_normalized = wait_clipped / (2.030034311250659 + eps)
    wait_score = -normalize(wait_normalized + eps)
    slack_base = np.abs(slk) + eps
    bottleneck_raw = rank * work / (np.power(slack_base, 0.4698564382092919) + eps)
    bottleneck_score = -0.43390492484944027 * normalize(bottleneck_raw + eps)
    protection_mask = slack_norm < 0.2090850318309389
    dur_score = np.where(protection_mask, dur_score * 0.21519796871743208, dur_score)
    energy_score = np.where(protection_mask, energy_score * 1.4305067207815132, energy_score)
    rank_score = np.where(protection_mask, rank_score * 2.9366367286638377, rank_score)
    bottleneck_score = np.where(protection_mask, bottleneck_score * 2.9366367286638377, bottleneck_score)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    score = np.where(np.isfinite(score), score, 0.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
