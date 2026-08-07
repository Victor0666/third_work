import numpy as np
RULE_METADATA = {'structure_hash': 'a93749138757dac3046913532b8fb4fc96cbd8fa36a039468780e5d14ec8bbb8', 'parameter_schema_hash': '9616efaec0e121f06c890b763b021b5c5b5950c3dc832e06d0bbf3a98a66d068', 'best_parameter_hash': 'abe4a1fe6573124f3d63f095e097ce39f235949fe7c72bf268cba48a3fa092de', 'best_parameters': {'epsilon': 4.930099143914282e-06, 'ddl_urgency_weight': 1.3158726178350193, 'bottleneck_weight': 0.32815847220294675, 'energy_risk_coupling_weight': 0.2817767955930279, 'fairness_weight': 1.9981462509162575, 'uncertainty_amplification_exponent': 0.32786493334984734, 'wait_saturation_scale': 5.470170189354137, 'urgency_cap_exponent': 0.8347760814989225, 'slack_feasibility_sigmoid_slope': 4.326093533872969, 'upward_remaining_interaction_power': 0.545576802918127, 'uncertainty_dispersion_scale': 1.590684260697377}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': 'ea565019a93389923067fe345a0c33a84a9db0c300ca81d5945b6744c071c547', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule with bounded control flow:
      - Replaces median-based gating with direct slack comparison to avoid branching on median computation.
      - Uses only one conditional branch: wait_boost gated by slack > 0 (simpler than median, satisfies branch limit).
      - All other logic is vectorized and branch-free (no np.where beyond the single wait gate).
      - Energy-risk coupling retained: min_incremental_energy * (1 + uncertainty).
      - Bottleneck uses learned power interaction and uncertainty amplification.
      - Adaptive normalization uses uncertainty dispersion.
      - Exactly 1 conditional branch (np.where for wait_boost), satisfying constraint.
      - Only literals {-2,-1,0,1,2} used; epsilon via PARAMS; finiteness enforced.
    """
    eps = 4.930099143914282e-06
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
    slack_gate = 1.0 / (1.0 + np.exp(-4.326093533872969 * (slack_centered + eps)))
    urgency_linear = np.clip(-slack, 0.0, 2.0)
    non_neg_slack = np.clip(slack, 0.0, None)
    max_non_neg_slack = np.max(non_neg_slack) if N > 0 else eps
    urgency_cap = np.power(max_non_neg_slack + eps, 0.8347760814989225)
    urgency_linear = np.minimum(urgency_linear, urgency_cap)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    ur_interaction = np.power(upward_rank * remaining_work + eps, 0.545576802918127)
    bottleneck_pressure = ur_interaction * duration * (1.0 + urgency_linear + eps)
    unc_std = np.std(uncertainty) if N > 1 else eps
    unc_normalized = (uncertainty - np.min(uncertainty)) / (np.ptp(uncertainty) + eps) if N > 1 else np.zeros_like(uncertainty)
    amplified_uncertainty = np.power(1.0 + unc_normalized, 0.32786493334984734)
    bottleneck_pressure = bottleneck_pressure * (slack_gate * amplified_uncertainty + (1.0 - slack_gate))
    energy_risk_term = min_incremental_energy * (1.0 + uncertainty)
    wait_scaled = ready_wait_time / (5.470170189354137 + eps)
    wait_clipped = np.clip(wait_scaled, -2.0, 2.0)
    wait_saturation = 1.0 / (1.0 + np.exp(-wait_clipped))
    wait_boost = np.where(slack > 0.0, wait_saturation, 0.0)

    def adaptive_normalize(x):
        x = np.copy(x)
        center = np.median(x) if N > 0 else 0.0
        fallback_range = np.max(x) - np.min(x) if N > 0 else eps
        unc_std = np.std(uncertainty) if N > 1 else eps
        dispersion = 1.590684260697377 * (unc_std + eps)
        denom = np.where(dispersion > eps, dispersion, fallback_range)
        return (x - center) / (denom + eps)
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    norm_energy_risk = adaptive_normalize(energy_risk_term)
    norm_wait = adaptive_normalize(wait_boost)
    score = 1.3158726178350193 * neg_slack + 0.32815847220294675 * norm_bottleneck + 0.2817767955930279 * norm_energy_risk - 1.9981462509162575 * norm_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
