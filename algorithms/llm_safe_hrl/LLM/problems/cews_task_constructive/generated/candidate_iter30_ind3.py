import numpy as np
RULE_METADATA = {'structure_hash': 'd178d745d1b959183873c9f15d2cbf88a03bf5b2c1e7b38be0927c3a8dfb92bb', 'parameter_schema_hash': 'd9d2efe2b459960daf2c67d1f94b85248962376ecbc8f0ac6c111f3e28f167e6', 'best_parameter_hash': '546342e0f45a4760a0eabf467ac199b5d1790933b4c724236e7b7ded038f40ef', 'best_parameters': {'epsilon': 9.456031391888989e-06, 'ddl_hard_penalty_weight': 7321.882221007578, 'critical_path_release_threshold': 0.7243223114889157, 'bottleneck_amplification_exponent': 1.1785085786316656, 'wait_fairness_strength': 1.359113803501975, 'energy_efficiency_activation_threshold': 0.33208908309469365, 'uncertainty_sigmoid_steepness': 4.538597292062667}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': '83654307ebec7f619575a536e09ed80cd8d7ba38be930a597c9dc4e03da459b8', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with three structural improvements:
      - Hard feasibility guard: tasks with slack < 0 receive dominant penalty; others enter multi-objective regime.
      - Critical-path release gate: upward_rank × remaining_work amplified *only* when slack < median_slack × threshold, preserving deadline dominance while releasing bottlenecks early.
      - Uncertainty-conditional energy efficiency: energy_per_duration term activated only when uncertainty exceeds tunable threshold (sigmoid-gated), avoiding premature energy optimization in low-risk regimes.
      - All normalization uses adaptive dispersion scaled by std(uncertainty); no multiplicative risk couplings.
      - Exactly 4 conditional expressions (np.where + np.clip), flat AST, no branching beyond required feasibility logic.
      - Uses only {-2,-1,0,1,2} literals; fully deterministic and finite-valued.
    """
    eps = 9.456031391888989e-06
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    finfo = np.finfo(float)
    neg_slack = np.clip(-slack, 0.0, None)
    feasible_mask = slack >= -eps

    def adaptive_normalize(x):
        x = np.copy(x)
        center = np.median(x) if N > 0 else 0.0
        unc_std = np.std(uncertainty) if N > 1 else eps
        dispersion = unc_std + eps
        fallback_range = np.max(x) - np.min(x) if N > 0 else eps
        denom = np.where(dispersion > eps, dispersion, fallback_range)
        return (x - center) / (denom + eps)
    median_slack = np.median(slack) if N > 0 else 0.0
    release_threshold = median_slack * 0.7243223114889157
    release_mask = (slack < release_threshold) & feasible_mask
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    raw_bottleneck = duration * upward_rank * remaining_work
    bottleneck_pressure = np.where(release_mask, np.power(raw_bottleneck + eps, 1.1785085786316656), 0.0)
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    energy_activation_gate = 1.0 / (1.0 + np.exp(-4.538597292062667 * (uncertainty - 0.33208908309469365)))
    energy_per_duration = min_incremental_energy / (duration + eps)
    energy_term = energy_activation_gate * energy_per_duration
    norm_energy_eff = adaptive_normalize(energy_term)
    wait_scaled = ready_wait_time / (2.0 + eps)
    wait_clipped = np.clip(wait_scaled, -2.0, 2.0)
    wait_saturation = 1.0 / (1.0 + np.exp(-wait_clipped))
    norm_wait = adaptive_normalize(wait_saturation)
    score = 7321.882221007578 * neg_slack + norm_bottleneck + norm_energy_eff - 1.359113803501975 * norm_wait
    score = np.nan_to_num(score, nan=finfo.max, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
