import numpy as np
RULE_METADATA = {'structure_hash': '1ad0f25bc3e337c966bd0258a0ffc57c9cdb0ddbfe183891b5fc0c06a5a96d73', 'parameter_schema_hash': 'fa4c60fde4e04844f9ff3b2149ecf9e785394ae70a708c36b97b7d365dbe3719', 'best_parameter_hash': '1de11c8a869623791beb1e97dd18ca01ab122d4716edd470cea984f75cc63358', 'best_parameters': {'epsilon': 0.04673820210200155, 'ddl_protection_gate_threshold': 0.6110808215724256, 'energy_duration_ratio_weight': 1.7953794098254663, 'successor_bottleneck_coupling': 0.3062516942592262, 'iqr_low_percentile': 39.81880072992741, 'iqr_high_percentile': 79.10710717295888, 'slack_sigmoid_steepness': 6.479635443072856, 'slack_sigmoid_offset': 0.9579289912989328, 'sigmoid_clip_bound': 22.54797518820373, 'wait_ramp_saturation_factor': 0.5749070814201701, 'urgency_bias_weight': 0.8438991419015821}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': '443ee56cc9efbb26b28dd5959a867bcf1ff1c9c789120ba1c12303ae52dde199', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining strengths from both parents:
    - Retains Parent 2's sharp sigmoid urgency with steepness/offset tuning for precise zero-slack discrimination.
    - Keeps adaptive wait ramp with saturation factor for fairness robustness in congestion.
    - Preserves remaining_work in bottleneck pressure to reflect downstream load impact.
    - Introduces novel 'urgency_bias_weight': scales normalized urgency *before* bottleneck coupling to enforce stronger deadline-first hierarchy.
    - Uses joint DDL-risk amplification (upward_rank × ddl_risk_condition) only under hard violation, avoiding over-prioritization.
    - All normalizations use unified adaptive IQR→min-max fallback; no unbounded ops or hidden state.
    """
    eps = 0.04673820210200155
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
        q_low = np.percentile(x, 39.81880072992741)
        q_high = np.percentile(x, 79.10710717295888)
        iqr = q_high - q_low
        center = np.median(x)
        if iqr < eps:
            x_min, x_max = (np.min(x), np.max(x))
            denom = x_max - x_min
            if denom < eps:
                return np.zeros_like(x)
            return (x - x_min) / (denom + eps)
        else:
            denom = iqr
            return (x - center) / (denom + eps)
    slack_centered = slack - 0.9579289912989328
    sigmoid_input = np.clip(-6.479635443072856 * slack_centered, -22.54797518820373, 22.54797518820373)
    urgency = 1.0 / (1.0 + np.exp(sigmoid_input))
    norm_urgency = adaptive_normalize(urgency)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy_eff = adaptive_normalize(energy_per_duration)
    bottleneck_pressure = duration * upward_rank * (1.0 + urgency + eps) * (1.0 + uncertainty + eps) * (remaining_work + eps)
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    if N == 1:
        median_wait = ready_wait_time[0]
    else:
        median_wait = np.median(ready_wait_time)
    saturation_wait = median_wait * 0.5749070814201701
    wait_ramp = np.clip(ready_wait_time / (saturation_wait + eps), 0.0, 1.0)
    norm_wait = adaptive_normalize(wait_ramp)
    max_uncertainty = np.max(uncertainty) if N > 0 else eps
    ddl_risk_condition = ((slack < 0.0) & (uncertainty > 0.6110808215724256 * max_uncertainty)).astype(float)
    ddl_risk_amplification = ddl_risk_condition * upward_rank
    biased_urgency = 0.8438991419015821 * norm_urgency
    score = biased_urgency + 0.3062516942592262 * norm_bottleneck + ddl_risk_amplification + 1.7953794098254663 * norm_energy_eff - norm_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
