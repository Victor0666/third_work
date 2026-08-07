import numpy as np
RULE_METADATA = {'structure_hash': '5ba077852163c3db73161c06f450256324caf66b442a9979a82eb240cc20f58d', 'parameter_schema_hash': '8b35e490a7ffcd1c128c093abe14c96a4f22e4ddbc085170863ef7897a5140a6', 'best_parameter_hash': 'a0a142a0bedc9eb68514da25e7806a63bc284820f4e6ed4a14a0aea521e6155b', 'best_parameters': {'epsilon': 1.1495826100143898e-05, 'ddl_urgency_weight': 0.7738562726913312, 'bottleneck_weight': 0.5946915059568725, 'energy_efficiency_weight': 0.24001256740687416, 'fairness_weight': 0.8832289317317352, 'uncertainty_amplification_exponent': 0.841024883610856, 'wait_saturation_scale': 4.158582354721637, 'urgency_cap_exponent': 0.9883822028522158, 'slack_feasibility_sigmoid_slope': 11.711086591238471, 'upward_remaining_interaction_power': 0.6160401267460507, 'uncertainty_dispersion_scale': 1.2240990480915794}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': '17ac7db271f69a6107b692758cc398503b094c55a79cb7a45f7165c843ea6aec', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with three structural improvements:
      - Replaces binary slack gate with smooth sigmoid gate using tunable slope for gradient-friendly CMA-ES optimization.
      - Restores core bottleneck interaction: upward_rank^p × remaining_work^p (p learned) — critical for preventing critical-path starvation.
      - Uses *all weights declared in PARAMETER_SCHEMA* (no hidden constants) — eliminates inactive-parameter fragility;
        all tunables now govern only gates, interactions, normalization, and final linear combination.
      - All declared parameters are used; no unused or missing PARAMS references.
      - Only {-2,-1,0,1,2} literals appear in function body.
    """
    eps = 1.1495826100143898e-05
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    neg_slack = np.clip(-slack, 0.0, None)
    slack_centered = -slack
    slack_gate = 1.0 / (1.0 + np.exp(-11.711086591238471 * (slack_centered + eps)))
    urgency_linear = np.clip(-slack, 0.0, 2.0)
    non_neg_slack = np.clip(slack, 0.0, None)
    max_non_neg_slack = np.max(non_neg_slack) if N > 0 else eps
    urgency_cap = np.power(max_non_neg_slack + eps, 0.9883822028522158)
    urgency_linear = np.minimum(urgency_linear, urgency_cap)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    ur_interaction = np.power(upward_rank * remaining_work + eps, 0.6160401267460507)
    bottleneck_pressure = ur_interaction * duration * (1.0 + urgency_linear + eps)
    unc_normalized = (uncertainty - np.min(uncertainty)) / (np.ptp(uncertainty) + eps) if N > 1 else np.zeros_like(uncertainty)
    amplified_uncertainty = np.power(1.0 + unc_normalized, 0.841024883610856)
    bottleneck_pressure = bottleneck_pressure * (slack_gate * amplified_uncertainty + (1.0 - slack_gate))

    def adaptive_normalize(x):
        x = np.copy(x)
        center = np.median(x) if N > 0 else 0.0
        unc_std = np.std(uncertainty) if N > 1 else eps
        dispersion = 1.2240990480915794 * (unc_std + eps)
        fallback_range = np.max(x) - np.min(x) if N > 0 else eps
        denom = np.where(dispersion > eps, dispersion, fallback_range)
        return (x - center) / (denom + eps)
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy_eff = adaptive_normalize(energy_per_duration)
    wait_scaled = ready_wait_time / (4.158582354721637 + eps)
    wait_clipped = np.clip(wait_scaled, -2.0, 2.0)
    wait_saturation = 1.0 / (1.0 + np.exp(-wait_clipped))
    norm_wait = adaptive_normalize(wait_saturation)
    score = 0.7738562726913312 * neg_slack + 0.5946915059568725 * norm_bottleneck + 0.24001256740687416 * norm_energy_eff - 0.8832289317317352 * norm_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
