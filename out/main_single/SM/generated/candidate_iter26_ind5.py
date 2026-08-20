import numpy as np
RULE_METADATA = {'structure_hash': 'f57eb5fbca64422f985f46312a45c363dabf10b02f923179f52d16b9383c5e5e', 'parameter_schema_hash': '52db11f1568990e3673e5458170f978f6785a1a1432f47b95b1c623715e4196b', 'best_parameter_hash': '6490b5d2d4b681f0f6cd3fd4e45f2d90e16595941c5b9245d6f6a97450ccc2ae', 'best_parameters': {'epsilon': 0.0002278388797334863, 'slack_risk_penalty': 1.8700686788710876, 'slack_urgency_gain': 3.3532484033596655, 'energy_efficiency_weight': 0.781109160105055, 'criticality_weight': 0.44963761965214155, 'bottleneck_proximity_weight': 1.9161299940619718, 'duration_uncertainty_ratio': 0.6079594580207333, 'wait_ramp_threshold': 95.19534796586113, 'ddl_protection_gate': 0.01336179024270869, 'finfo_max_scale': 252316.9081066996, 'robust_normalization_quantile': 0.7009284690394566, 'successor_release_sharpness': 3.055089513130739}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': '8f04a113feca612c7d31ce1175326d4c906e22a0ef84276d49f58e81102a0850', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining strengths of both parents:
      - Keeps Parent 2's robust sigmoid ddl_protection_gate and slack_urgency_gain.
      - Adds Parent 1's *additive* slack-coupled energy modulation (not multiplicative) to avoid blowup near zero slack.
      - Retains Parent 2's simplified linear duration-uncertainty blend but augments with Parent 1's starvation-aware wait_ramp.
      - Introduces new 'energy_slack_coupling_strength' via reuse of existing 'ddl_protection_gate' — avoids adding parameter.
        Specifically: use PARAMS["ddl_protection_gate"] as coupling strength (same semantic role: controls sensitivity near slack=0).
      - All normalizations use same robust quantile-based scheme; no hard clipping or median bias.
      - Strict DDL-first enforced via dominant slack_penalty term.
    """
    eps = 0.0002278388797334863
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
            scale = np.quantile(abs_x[finite_mask], 0.7009284690394566)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(0.01336179024270869 * slack_norm))
    slack_penalty = np.where(slk < 0, 1.8700686788710876 * np.abs(slack_norm), -3.3532484033596655 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    base_energy_score = -0.781109160105055 * normalize(inv_energy)
    slack_proximity = np.clip(1.0 - np.abs(slack_norm), 0.0, 1.0)
    energy_score = base_energy_score * ddl_gate + 0.01336179024270869 * base_energy_score * slack_proximity
    rank_score = -0.44963761965214155 * ddl_gate * normalize(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 3.055089513130739)
    bottleneck_score = -1.9161299940619718 * normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.6079594580207333 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 95.19534796586113)
    wait_normalized = normalize(wait_clipped + eps)
    wait_ramp = np.clip(wait_normalized, 0.0, 1.0)
    wait_score = -wait_ramp
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 252316.9081066996
    min_safe = -finfo.max / 252316.9081066996
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
