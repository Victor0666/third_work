import numpy as np
RULE_METADATA = {'structure_hash': '6fae201926f2e2129899abe6db3eccab3208d9e334e56bcf10d839fe27a15514', 'parameter_schema_hash': 'd9d51bd7f8ff96e288ba771a76285c36d58322b4ae2b65a5a076f2d118b50b6d', 'best_parameter_hash': 'cd9a25a91f8ee0a7f9fd4a640f39bc9cf54a5b1561b59f3a545eccfbfa760062', 'best_parameters': {'epsilon': 3.77477677729143e-05, 'ddl_risk_gate_weight': 2.965723934112103, 'bottleneck_uncertainty_amplification': 1.031550713104168, 'uncertainty_dispersion_scale': 1.612876480867119, 'wait_fairness_weight': 0.848249170256851, 'upward_rank_remaining_work_coupling': 0.6774944273712066, 'successor_slack_deficit_weight': 1.1802406847338993, 'ddl_risk_activation_threshold': 1.0469258138278676, 'urgency_tanh_scale': 0.8449481173472997, 'host_load_energy_gate_threshold': 0.582487070926299, 'energy_efficiency_coupling_strength': 0.7149772242417761}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': 'd5a4a47cda225d458717edece1239269c0f816ed9837e5cb0a4e23e957171098', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with three structural improvements:
      - Host-load–conditional energy gating: suppresses marginal energy penalty when host load is low (avoids unnecessary energy cost).
      - Urgency-successor-coupled energy modulation: scales energy efficiency term by urgency * successor_risk_proxy to focus savings where it matters most.
      - Reintroduced bounded tanh urgency (replacing exponentiated form) for robust deadline sensitivity without instability.
      - All other components preserved: adaptive normalization, tightened DDL gate, fairness, bottleneck pressure, and successor proxy.
      - No numeric literals beyond {-2,-1,0,1,2}; all tunables exposed via PARAMS.
    """
    eps = 3.77477677729143e-05
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
        if N == 0:
            return np.zeros(0, dtype=float)
        if N == 1:
            return np.zeros(1, dtype=float)
        x_min, x_max = (np.min(x), np.max(x))
        range_val = x_max - x_min
        unc_median = np.median(uncertainty) if N > 0 else eps
        dispersion = 1.612876480867119 * (unc_median + eps)
        denom = np.where(range_val > eps, range_val, dispersion + eps)
        return (x - x_min) / (denom + eps)
    neg_slack = np.clip(-slack, 0.0, None)
    unc_median = np.median(uncertainty) if N > 0 else eps
    ddl_risk_condition = (slack < 0) & (uncertainty > unc_median * 1.0469258138278676 + eps)
    ddl_risk_gate = np.where(ddl_risk_condition, 1.0, 0.0)
    median_slack = np.median(slack) if N > 0 else 0.0
    urgency_base = np.clip(median_slack - slack, 0.0, 2.0)
    urgency = np.tanh(0.8449481173472997 * urgency_base)
    norm_wait = adaptive_normalize(ready_wait_time)
    bottleneck_pressure = remaining_work * upward_rank * 0.6774944273712066 * (1.0 + norm_wait * (slack < 0).astype(float))
    norm_uncertainty = adaptive_normalize(uncertainty)
    bottleneck_pressure = bottleneck_pressure * np.power(1.0 + norm_uncertainty, 1.031550713104168)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy_eff = adaptive_normalize(energy_per_duration)
    fairness_term = -0.848249170256851 * norm_wait
    successor_risk_proxy = np.where((slack < median_slack - eps) & (uncertainty > unc_median), 1.0, 0.0)
    slack_surplus = np.clip(slack, 0.0, None)
    norm_slack_surplus = adaptive_normalize(slack_surplus)
    system_headroom_proxy = np.sqrt((norm_uncertainty + eps) * (norm_slack_surplus + eps))
    host_load_energy_gate = (system_headroom_proxy <= 0.582487070926299).astype(float)
    energy_modulation = urgency * successor_risk_proxy * 0.7149772242417761
    gated_energy_term = norm_energy_eff * host_load_energy_gate * energy_modulation
    score = neg_slack + 2.965723934112103 * ddl_risk_gate + urgency + adaptive_normalize(bottleneck_pressure) + gated_energy_term + fairness_term + 1.1802406847338993 * successor_risk_proxy
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
