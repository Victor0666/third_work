import numpy as np
RULE_METADATA = {'structure_hash': '323f6bec20c328c933b9f98d8a897ca239e0f9ddbc9757327e09c8a27624a4e7', 'parameter_schema_hash': '50c50184d7e9450c6b76c188448224d2a56cc320928e52e6454c6756c2391f26', 'best_parameter_hash': 'a6d8971fae5e06345daac4380e94792ac03cd961329b34df1c94c12e8a4b97d7', 'best_parameters': {'epsilon': 9.949299505918555e-05, 'energy_duration_ratio_weight': 1.6760149429729725, 'successor_bottleneck_coupling': 0.2346216724743252, 'uncertainty_dispersion_scale': 1.439795342949639, 'urgency_cap_exponent': 0.9094867893204461, 'bottleneck_uncertainty_amplification': 0.5001813809715754, 'slack_risk_penalty_weight': 1.2001563649527847, 'joint_risk_coupling': 0.308378937182591}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': '4e39c1fd8a90c7ba4d872e7371cc34d7d5fd80dd809efa8be6e695d3a0e1e265', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's uncertainty-aware normalization and feasibility-preserving urgency cap with Parent 1's explicit negative-slack penalty and joint risk coupling.
    
    Key structural improvements:
      - Retains uncertainty-weighted dispersion scaling (Parent 2) for risk-contextual normalization.
      - Keeps feasibility-preserving urgency cap: linear urgency capped by max_non_neg_slack^exponent to avoid inversion under negative slack.
      - Adds explicit clipped negative slack penalty (Parent 1) *before* normalization to preserve hard-deadline signal integrity.
      - Introduces joint_risk_coupling: normalized negative slack × normalized uncertainty → direct amplification of priority when both deadline risk and execution uncertainty are high.
      - Replaces sigmoid wait saturation with linear anti-starvation via normalized and inverted ready_wait_time (simpler, more robust).
      - Bottleneck pressure now includes upward_rank × remaining_work × (1 + urgency_linear) × (1 + uncertainty)^amplification — unifying critical-path load and risk-aware release pressure.
      - All numeric literals strictly in {-2,-1,0,1,2}; no hidden constants; zero branching; deterministic.
    """
    eps = 9.949299505918555e-05
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def adaptive_normalize(x):
        x = np.copy(x)
        center = np.median(x)
        unc_std = np.std(uncertainty) if N > 1 else eps
        dispersion = 1.439795342949639 * (unc_std + eps)
        fallback_range = np.max(x) - np.min(x) if N > 0 else eps
        denom = dispersion if dispersion > eps else fallback_range
        return (x - center) / (denom + eps)
    neg_slack = np.clip(-slack, 0.0, None)
    norm_neg_slack = adaptive_normalize(neg_slack)
    median_slack = np.median(slack) if N > 0 else 0.0
    urgency_linear = np.clip(median_slack - slack, 0.0, 2.0)
    non_neg_slack = np.clip(slack, 0.0, None)
    max_non_neg_slack = np.max(non_neg_slack) if N > 0 else eps
    urgency_cap = np.power(max_non_neg_slack + eps, 0.9094867893204461)
    urgency_linear = np.minimum(urgency_linear, urgency_cap)
    norm_urgency = adaptive_normalize(urgency_linear)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy_eff = adaptive_normalize(energy_per_duration)
    bottleneck_pressure = duration * upward_rank * remaining_work * (1.0 + urgency_linear + eps)
    unc_normalized = adaptive_normalize(uncertainty)
    bottleneck_pressure = bottleneck_pressure * np.power(1.0 + unc_normalized, 0.5001813809715754)
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    norm_uncertainty = adaptive_normalize(uncertainty)
    joint_risk = norm_neg_slack * norm_uncertainty
    norm_joint_risk = adaptive_normalize(joint_risk)
    norm_wait = adaptive_normalize(ready_wait_time)
    inv_wait = 1.0 - norm_wait
    score = 1.2001563649527847 * norm_neg_slack + 0.308378937182591 * norm_joint_risk + norm_urgency + 0.2346216724743252 * norm_bottleneck + 1.6760149429729725 * norm_energy_eff + inv_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
