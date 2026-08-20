import numpy as np
RULE_METADATA = {'structure_hash': '65584644bfb32d557e0707834e280a0f73114a2ce1f7909e36a9123af145b991', 'parameter_schema_hash': '24aad060bb0eca1646e0ec2ecb0e5e7b360ffdf346f37c60dae4de77bf0a4c58', 'best_parameter_hash': 'eb199f5eaba8cb5b14a96cdb5578cb87bf56430689d6087682d71e20c65d16e0', 'best_parameters': {'epsilon': 0.012359879739959237, 'slack_risk_penalty': 1.810225700046763, 'slack_urgency_gain': 4.283396119249236, 'energy_efficiency_weight': 3.037349496837924, 'criticality_weight': 1.0143832853138284, 'bottleneck_proximity_weight': 0.11833909001553436, 'duration_uncertainty_ratio': 1.291299917676048, 'wait_ramp_threshold': 12.684286428088171, 'ddl_feasibility_margin': 0.1167627623844163, 'robust_normalization_quantile': 0.6966557138024614, 'successor_release_sharpness': 3.315023327488025, 'score_clipping_bound': 3325234817.709443}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': 'b8cda474cad2003276df97b5b0c9f9b1a335f4a02f13ecfb83b4dc6a14e53f96', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule with 12 parameters:
      - Retains Parent 2's piecewise-linear DDL feasibility margin and semantic-aware normalization.
      - Replaces Parent 1's fragile sigmoid gate and exponentiated ddl_gate with direct slack-magnitude coupling via |slack|^{-exponent}, but reuses `successor_release_sharpness` parameter to avoid adding a new one.
      - Uses `successor_release_sharpness` for both bottleneck sharpening *and* energy slack coupling — justified because both benefit from similar deadline-tightness sensitivity.
      - All numeric literals are -2,-1,0,1,2; epsilon and clipping use PARAMS; no hidden constants.
      - Final score remains dominated by slack penalty to enforce hard deadline adherence first.
    """
    eps = 0.012359879739959237
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def normalize_abs_quantile(x, q=0.6966557138024614):
        x = np.asarray(x)
        abs_x = np.abs(x)
        finite_mask = np.isfinite(abs_x)
        if np.any(finite_mask):
            scale = np.quantile(abs_x[finite_mask], q)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)

    def normalize_pos_quantile(x, q=0.6966557138024614):
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
    slack_penalty = np.where(slk < 0, 1.810225700046763 * np.abs(slack_norm), -4.283396119249236 * np.abs(slack_norm))
    margin = 0.1167627623844163
    ddl_feasible_mask = np.clip((slk + margin) / (2 * margin + eps), 0.0, 1.0)
    slack_magnitude = np.abs(slk) + eps
    slack_coupling_factor = np.power(slack_magnitude, -3.315023327488025)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -3.037349496837924 * normalize_pos_quantile(inv_energy) * ddl_feasible_mask * slack_coupling_factor
    rank_score = -1.0143832853138284 * normalize_pos_quantile(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 3.315023327488025)
    bottleneck_score = -0.11833909001553436 * normalize_pos_quantile(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize_abs_quantile(duration + eps)
    uncert_norm = normalize_abs_quantile(uncert + eps)
    dur_uncert_blend = dur_norm + 1.291299917676048 * uncert_norm
    dur_score = normalize_abs_quantile(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 12.684286428088171)
    wait_normalized = normalize_pos_quantile(wait_clipped + eps)
    wait_ramp = np.clip(wait_normalized, 0.0, 1.0)
    wait_score = -wait_ramp
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    score = np.nan_to_num(score, nan=0.0, posinf=3325234817.709443, neginf=-3325234817.709443)
    score = np.clip(score, -3325234817.709443, 3325234817.709443)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
