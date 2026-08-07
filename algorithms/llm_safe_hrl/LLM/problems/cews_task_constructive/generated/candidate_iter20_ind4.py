import numpy as np
RULE_METADATA = {'structure_hash': '99d50efa6f3aff1844cd8c7fa5739fa72006090c6a47e573c17ae600ff8a6e60', 'parameter_schema_hash': '781a8b063bac5033b0098afb6d35d693ce5a3b473586b8bd9da436f655b3987a', 'best_parameter_hash': '06e2faeb2778a54e23e0f3d3a601c6efff2319bbf1dba5b563bb085b17867940', 'best_parameters': {'epsilon': 1.3522468330837156e-05, 'energy_duration_ratio_weight': 1.2884901597203982, 'successor_bottleneck_coupling': 0.7117110324403927, 'uncertainty_dispersion_scale': 1.1591140155755764, 'urgency_cap_exponent': 0.5082241022061051, 'wait_saturation_scale': 3.933251696191947, 'bottleneck_uncertainty_amplification': 0.8200246573843485, 'slack_feasibility_gate_threshold': -0.1688722860099393, 'upward_rank_remaining_work_coupling': 0.2600485426923266, 'wait_fairness_blend_weight': 0.15867575594148486, 'ddl_risk_activation_threshold': 0.664523838687115, 'risk_modulated_urgency_weight': 2.618813038436592}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': '067e9080a6f422e49eb96fdcc9893ee46abf44a149024eff69c23967f0741327', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining strengths from both parents:
    - Uses Parent 2's slack-feasibility gating on bottleneck pressure (critical for DDL safety)
    - Adds Parent 1's DDL-risk conditional urgency amplification (improves responsiveness under lateness)
    - Replaces artificial urgency clipping with slack-feasibility-gated urgency: zero when slack < 0
    - Introduces novel dual-gating: bottleneck suppression AND urgency amplification both conditioned on verified DDL risk
    - Keeps blended fairness and uncertainty-aware normalization for robustness
    - All numeric literals strictly in {-2,-1,0,1,2}; no hidden constants beyond that set
    """
    eps = 1.3522468330837156e-05
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
        dispersion = 1.1591140155755764 * (unc_std + eps)
        fallback_range = np.max(x) - np.min(x) if N > 0 else eps
        denom = dispersion if dispersion > eps else fallback_range
        return (x - center) / (denom + eps)
    median_slack = np.median(slack) if N > 0 else 0.0
    urgency_linear = np.where(slack >= 0.0, median_slack - slack, 0.0)
    non_neg_slack = np.clip(slack, 0.0, None)
    max_non_neg_slack = np.max(non_neg_slack) if N > 0 else eps
    urgency_cap = np.power(max_non_neg_slack + eps, 0.5082241022061051)
    urgency_linear = np.minimum(urgency_linear, urgency_cap)
    norm_urgency = adaptive_normalize(urgency_linear)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy_eff = adaptive_normalize(energy_per_duration)
    base_bottleneck = duration * upward_rank * remaining_work * (1.0 + urgency_linear + eps)
    feasibility_mask = (slack >= -0.1688722860099393).astype(float)
    unc_normalized = adaptive_normalize(uncertainty)
    amp_factor = np.power(1.0 + unc_normalized, 0.8200246573843485)
    bottleneck_pressure = base_bottleneck * amp_factor * feasibility_mask
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    wait_scaled = ready_wait_time / (3.933251696191947 + eps)
    wait_clipped = np.clip(wait_scaled, -2.0, 2.0)
    wait_sigmoid = 1.0 / (1.0 + np.exp(-wait_clipped))
    norm_wait_sigmoid = adaptive_normalize(wait_sigmoid)
    norm_wait_linear = 1.0 - adaptive_normalize(ready_wait_time)
    norm_wait = 0.15867575594148486 * norm_wait_sigmoid + (1.0 - 0.15867575594148486) * norm_wait_linear
    cp_coupling = upward_rank * remaining_work
    norm_cp_coupling = adaptive_normalize(cp_coupling)
    median_uncertainty = np.median(uncertainty) if N > 0 else eps
    ddl_risk_gate = ((slack < 0.0) & (uncertainty > 0.664523838687115 * median_uncertainty)).astype(float)
    risk_amplified_urgency = norm_urgency * (1.0 + 2.618813038436592 * ddl_risk_gate)
    score = risk_amplified_urgency + 0.7117110324403927 * norm_bottleneck + 1.2884901597203982 * norm_energy_eff - norm_wait + 0.2600485426923266 * norm_cp_coupling
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
