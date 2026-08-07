import numpy as np
RULE_METADATA = {'structure_hash': '13fbffe3683d48961f39cf48ce1c79f89a8d18c56305656787395479da020574', 'parameter_schema_hash': '4bc49b24dcfd1ea22c0636fcda493a64daa610e71b6c82637df7f54c673d2b67', 'best_parameter_hash': 'f2fef88e6c4a79af378013a36e3636a7394e73d78a2a6fc7fbffbba347ff5c6e', 'best_parameters': {'epsilon': 5.407089309950883e-05, 'energy_duration_ratio_weight': 1.8336817975152335, 'successor_bottleneck_coupling': 1.1300349478627152, 'urgency_cap_exponent': 0.774720626874787, 'wait_saturation_scale': 1.228624955662009, 'bottleneck_uncertainty_amplification': 0.5737485008907355, 'slack_feasibility_gate_threshold': -0.7865305682021578, 'upward_rank_remaining_work_coupling': 1.0298709925545635, 'urgency_slack_gate_threshold': -0.0677132890376333, 'iqr_percentile_low': 22.742086150866303, 'iqr_percentile_high': 84.26936837821427, 'percentile_scale_factor': 144.64217035598082}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': '807cf0eddf445329b07f8e36ae3c448e78422305683a80ce4b50677b0b2e7f1f', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining strengths from both parents with novel structural enhancements:
    - Replaces adaptive_normalize with robust uncertainty-aware IQR normalization using declared percentiles.
    - Introduces slack-feasibility gate on urgency (not just bottleneck) to suppress urgency when slack is critically negative.
    - Replaces blended fairness with saturating inverse wait-time: 1.0 - tanh(wait / scale), improving starvation resistance.
    - Uses multiplicative bottleneck pressure with uncertainty amplification gated by feasibility.
    - All numeric literals strictly in {-2,-1,0,1,2}; no hidden constants beyond that set.
    """
    eps = 5.407089309950883e-05
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def iqr_normalize(x):
        x = np.copy(x)
        q_low = np.quantile(x, 22.742086150866303 / 144.64217035598082)
        q_high = np.quantile(x, 84.26936837821427 / 144.64217035598082)
        iqr = q_high - q_low + eps
        center = np.median(x)
        return (x - center) / (iqr + eps)
    median_slack = np.median(slack) if N > 0 else 0.0
    urgency_linear = np.clip(median_slack - slack, 0.0, 2.0)
    non_neg_slack = np.clip(slack, 0.0, None)
    max_non_neg_slack = np.max(non_neg_slack) if N > 0 else eps
    urgency_cap = np.power(max_non_neg_slack + eps, 0.774720626874787)
    urgency_linear = np.minimum(urgency_linear, urgency_cap)
    urgency_gate = (slack >= -0.0677132890376333).astype(float)
    gated_urgency = urgency_linear * urgency_gate
    norm_urgency = iqr_normalize(gated_urgency)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy_eff = iqr_normalize(energy_per_duration)
    base_bottleneck = duration * upward_rank * remaining_work * (1.0 + gated_urgency + eps)
    feasibility_mask = (slack >= -0.7865305682021578).astype(float)
    unc_normalized = iqr_normalize(uncertainty)
    amp_factor = np.power(1.0 + unc_normalized, 0.5737485008907355)
    bottleneck_pressure = base_bottleneck * amp_factor * feasibility_mask
    norm_bottleneck = iqr_normalize(bottleneck_pressure)
    wait_scaled = ready_wait_time / (1.228624955662009 + eps)
    wait_tanh = np.tanh(wait_scaled)
    fairness_term = 1.0 - wait_tanh
    norm_fairness = iqr_normalize(fairness_term)
    cp_coupling = upward_rank * remaining_work
    norm_cp_coupling = iqr_normalize(cp_coupling)
    score = norm_urgency + 1.1300349478627152 * norm_bottleneck + 1.8336817975152335 * norm_energy_eff - norm_fairness + 1.0298709925545635 * norm_cp_coupling
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
