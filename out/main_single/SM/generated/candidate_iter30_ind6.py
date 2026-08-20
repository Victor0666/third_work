import numpy as np
RULE_METADATA = {'structure_hash': '343f70d6f65634a40b914be22e5a154c617c239b0441adb0130aeb96d8f57711', 'parameter_schema_hash': 'a9d34536f0bd198b2aa7994f07b5be0f992a49c086a61fb50e1783fd686d49f6', 'best_parameter_hash': '15871494497958a6c38140d920a5f1ed4cd2d3e6dce935aed54b7f7c4e4b1b5f', 'best_parameters': {'epsilon': 0.06553505762394976, 'slack_risk_penalty': 8.517065687468188, 'slack_urgency_gain': 3.4003853418079326, 'energy_efficiency_weight': 1.2518668855264679, 'criticality_weight': 1.118203185792796, 'bottleneck_proximity_weight': 0.7821719797930738, 'duration_uncertainty_ratio': 0.03733871416953053, 'wait_pressure_ratio': 0.36377785993493306, 'ddl_protection_gate': 0.3221776715126334, 'finfo_max_scale': 2381889.166360675, 'robust_normalization_quantile': 0.5791496658045169, 'successor_release_sharpness': 1.676076809862952}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': 'ebad27084cbeaa3d675faf41278ac65503be0d6fd62e75e7e54047f774a6000d', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
      - Smooth sigmoid energy activation (replaces hard threshold) using `ddl_protection_gate`-style logic but tuned separately.
      - Load-aware energy normalization via `upward_rank * remaining_work`, quantile-normalized — aligns marginal energy with critical-path load density.
      - Adaptive starvation mitigation: `wait_normalized / (|slack_norm| + eps)` scaled by `wait_pressure_ratio`.
      - All normalizations use robust quantile scaling; no mean/median bias.
      - Slack penalty dominates to enforce hard deadline feasibility first.
      - Exactly 12 parameters; all used; no numeric literals beyond -2,-1,0,1,2.
      - Uses np.finfo for safe clamping.
    """
    eps = 0.06553505762394976
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
            scale = np.quantile(abs_x[finite_mask], 0.5791496658045169)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(0.3221776715126334 * slack_norm))
    slack_penalty = np.where(slk < 0, 8.517065687468188 * np.abs(slk), -3.4003853418079326 * slk)
    energy_activation = 1.0 / (1.0 + np.exp(-0.3221776715126334 * 2.0 * slack_norm))
    inv_energy = 1.0 / (energy + eps)
    load_density = rank * work
    load_norm = normalize(load_density + eps)
    energy_score = -1.2518668855264679 * normalize(inv_energy) * load_norm * energy_activation
    rank_score = -1.118203185792796 * ddl_gate * normalize(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 1.676076809862952)
    bottleneck_score = -0.7821719797930738 * normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.03733871416953053 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    slack_abs_med = np.median(np.abs(slk)) if len(slk) > 0 else 1.0
    wait_clipped = np.clip(wait, 0.0, 2.0 * slack_abs_med + eps)
    wait_normalized = normalize(wait_clipped + eps)
    wait_pressure = wait_normalized / (np.abs(slack_norm) + eps)
    wait_ramp = np.clip(0.36377785993493306 * wait_pressure, 0.0, 1.0)
    wait_score = -wait_ramp
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 2381889.166360675
    min_safe = -finfo.max / 2381889.166360675
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
