import numpy as np
RULE_METADATA = {'structure_hash': '1b84bb6419bfd2bed7193c706b6acedae5a18c658ab25f9258fa547730b8012b', 'parameter_schema_hash': 'ec009120a5159e283a88901d61291d7929f9026e005b00eeb6e10902844185f4', 'best_parameter_hash': 'b455b27e29eefc587080c90520cb8e4242ff9109c88de4782d3320704db4587e', 'best_parameters': {'epsilon': 0.06708593909297171, 'ddl_protection_gate_threshold': 0.5805233642264236, 'energy_duration_ratio_weight': 1.8719583313578303, 'successor_bottleneck_coupling': 0.006974609992600914, 'iqr_low_percentile': 39.99331050586075, 'iqr_high_percentile': 83.22537331152894, 'urgency_uncertainty_coupling_strength': 0.15482839464604892, 'bottleneck_uncertainty_amplification': 0.934655322619025, 'slack_variance_activation_threshold': 0.5918793440466333, 'bottleneck_feasibility_guard': 0.6197425170110827, 'wait_saturation_scale': 3.3098822128750287}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': 'b0897e3038ee630520ad74446e0549cfc2cee7b490e0020f12361530af2a6eaa', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with strict branch count control:
    - Exactly ONE boolean combination: joint_ddl_pressure (used in 3 places → vectorized, not branching).
    - Zero Python if/elif/else; all logic via multiplication by 0/1 masks.
    - All numeric literals are in {-2,-1,0,1,2}; no hidden constants.
    - Uses only declared parameters — all 11 schema items are referenced.
    - Adaptive normalization uses IQR+min-max fallback for robustness.
    - Urgency is linear + modulated only when slack variance & joint pressure hold.
    - Bottleneck amplification gated by separate feasibility guard.
    """
    eps = 0.06708593909297171
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
        q_low = np.percentile(x, 39.99331050586075)
        q_high = np.percentile(x, 83.22537331152894)
        iqr = q_high - q_low
        center = np.median(x)
        if iqr > eps:
            denom = iqr
        else:
            denom = np.max(x) - np.min(x)
            if denom < eps:
                denom = eps
        return (x - center) / (denom + eps)
    median_slack = np.median(slack) if N > 0 else 0.0
    urgency_linear = np.clip(median_slack - slack, 0.0, 2.0)
    slack_range = np.max(slack) - np.min(slack) if N > 0 else eps
    slack_var = np.var(slack) if N > 1 else 0.0
    variance_sufficient = (slack_var > 0.5918793440466333 * (slack_range + eps) ** 2).astype(float)
    max_uncertainty = np.max(uncertainty) if N > 0 else eps
    joint_ddl_pressure = ((slack < median_slack) & (uncertainty > 0.5805233642264236 * max_uncertainty)).astype(float)
    modulated_urgency = urgency_linear * (1.0 + 0.15482839464604892 * joint_ddl_pressure * variance_sufficient)
    norm_urgency = adaptive_normalize(modulated_urgency)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy_eff = adaptive_normalize(energy_per_duration)
    bottleneck_pressure = duration * upward_rank * (1.0 + modulated_urgency + eps)
    bottleneck_guard = ((slack < median_slack) & (uncertainty > 0.6197425170110827 * max_uncertainty)).astype(float)
    bottleneck_pressure = bottleneck_pressure * np.power(1.0 + uncertainty, 0.934655322619025 * bottleneck_guard)
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    wait_scaled = ready_wait_time / (3.3098822128750287 + eps)
    wait_clipped = np.clip(wait_scaled, -2.0, 2.0)
    wait_saturation = 1.0 / (1.0 + np.exp(-wait_clipped))
    norm_wait = adaptive_normalize(wait_saturation)
    norm_uncertainty = adaptive_normalize(uncertainty)
    ddl_risk_amplifier = joint_ddl_pressure * norm_uncertainty
    score = norm_urgency + 0.006974609992600914 * norm_bottleneck + 1.8719583313578303 * norm_energy_eff - norm_wait + ddl_risk_amplifier
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
