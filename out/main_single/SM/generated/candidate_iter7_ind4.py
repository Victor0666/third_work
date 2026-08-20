import numpy as np
RULE_METADATA = {'structure_hash': '19fbab5630a085ec3a96ff223d874754f8b0b1f3e53e73cafadeec5e1a5e2a6f', 'parameter_schema_hash': '66ed5f6bae1022fa67e8c6d019de8771ac7f61c19999dc35a1f0a36aec8bd45b', 'best_parameter_hash': '2f60aeac68e227c15d7c51d2a6f0be1eb15183c555d3bd0fcd50a8117cb30bc2', 'best_parameters': {'epsilon': 3.884923704838727e-05, 'slack_risk_penalty': 6.115348086442496, 'slack_urgency_gain': 3.112599869394244, 'energy_efficiency_weight': 1.282728978850582, 'bottleneck_proximity_weight': 1.3269789947223718, 'duration_uncertainty_ratio': 1.1264077778847046, 'wait_clip_threshold': 2.190441596613292, 'finfo_max_scale': 7940766.001573785}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': '95a2c20b281011e7ecc8ee0235720015229f34a92ef2709a340b1b8e1262758e', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with:
      - Slack gating applied to *bottleneck-proximal criticality* (upward_rank * remaining_work)
      - Ready-wait time capped by bounded linear clipping instead of exponential saturation
      - Removed fragile uncertainty-slack interaction and inactive criticality_weight
      - Added robust median-based normalization for improved outlier resilience
      - Enforced hard DDL-first feasibility via slack sign gating before energy/criticality terms
      - All features scaled compatibly via median absolute scaling (more stable than mean-abs)
    """
    eps = 3.884923704838727e-05
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
        scale = np.median(abs_x) if np.all(np.isfinite(abs_x)) and np.median(abs_x) > eps else eps
        return x / (scale + eps)
    slack_feasible_mask = (slk >= 0).astype(np.float64)
    bottleneck_score = rank * work
    bottleneck_score_gated = bottleneck_score * slack_feasible_mask
    bottleneck_norm = normalize(bottleneck_score_gated + eps)
    criticality_score = -1.3269789947223718 * bottleneck_norm
    slack_penalty = np.where(slk < 0, 6.115348086442496 * np.abs(normalize(slk)), -3.112599869394244 * np.abs(normalize(slk)))
    inv_energy = 1.0 / (energy + eps)
    energy_norm = normalize(inv_energy)
    energy_score = -1.282728978850582 * energy_norm * slack_feasible_mask
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 1.1264077778847046 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 2.190441596613292)
    wait_score = -normalize(wait_clipped + eps)
    score = slack_penalty + criticality_score + energy_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 7940766.001573785
    min_safe = -finfo.max / 7940766.001573785
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
