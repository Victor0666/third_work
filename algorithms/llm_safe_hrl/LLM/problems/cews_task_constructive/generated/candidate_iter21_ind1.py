import numpy as np
RULE_METADATA = {'structure_hash': '556576e45552e1fd54e7e9ac21c30a748235e10a4d61c3ade8f901f91f294f16', 'parameter_schema_hash': '33dbdba4ea6d79a24c55602cb572446876be12729979fdb151afeed16b6bee9f', 'best_parameter_hash': '3349022f808efe550b31d847bd7d9abedc23a0cf1281517e04df63f1da3129ea', 'best_parameters': {'epsilon': 2.8796810711984273e-05, 'ddl_risk_activation_threshold': 0.703611618109566, 'successor_bottleneck_coupling': 0.3753422613967166, 'urgency_scale': 0.21072369591220025, 'urgency_clamp_lower': -0.7551093526002026, 'risk_modulated_bottleneck_weight': 1.1728290135370019, 'iqr_percentile_low': 24.193693761797967, 'iqr_percentile_high': 84.34853603919223, 'risk_gated_energy_weight': 0.6369300681615266}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': 'f986263aca79ccf190fe64fd14189355b6050386bf5be9c346c917a78a60788a', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
    - Removed inactive fairness parameters (wait_fairness_weight, ddl_risk_activation_threshold no longer used in fairness)
    - Reintroduced *multiplicative risk-gated energy term*: only activates under verified DDL risk (slack < 0 ∧ high uncertainty)
    - Restored clean bottleneck structure: duration × upward_rank × energy (no redundant modulations)
    - Unified risk gating: single ddl_risk_gate controls both bottleneck amplification AND energy penalty
    - Simplified normalization: removed redundant adaptive_normalize calls; reused same robust center/IQR logic
    - Strict finiteness: all divisions guarded, nan/inf replaced via np.finfo, no hidden constants beyond {-2,-1,0,1,2}
    """
    eps = 2.8796810711984273e-05
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
        q_low = np.percentile(x, 24.193693761797967)
        q_high = np.percentile(x, 84.34853603919223)
        iqr = q_high - q_low + eps
        center = np.median(x)
        return (x - center) / (iqr + eps)
    abs_slack = np.abs(slack)
    scale = np.median(abs_slack) if np.any(abs_slack > eps) else 1.0
    tanh_urgency = np.tanh(-slack / (scale * 0.21072369591220025 + eps))
    clamped_urgency = np.clip(tanh_urgency, -0.7551093526002026, 1.0)
    urgency = (clamped_urgency + 1.0) / 2.0
    norm_urgency = adaptive_normalize(urgency)
    duration = min_exec_time + min_comm_time
    bottleneck_pressure = duration * upward_rank * (min_incremental_energy + eps)
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    median_uncertainty = np.median(uncertainty) if N > 0 else eps
    ddl_risk_gate = ((slack < 0.0) & (uncertainty > 0.703611618109566 * median_uncertainty)).astype(float)
    risk_amplified_bottleneck = norm_bottleneck * (1.0 + 1.1728290135370019 * ddl_risk_gate)
    norm_energy = adaptive_normalize(min_incremental_energy)
    risk_gated_energy_penalty = 0.6369300681615266 * norm_energy * ddl_risk_gate
    score = norm_urgency + 0.3753422613967166 * risk_amplified_bottleneck + risk_gated_energy_penalty
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
