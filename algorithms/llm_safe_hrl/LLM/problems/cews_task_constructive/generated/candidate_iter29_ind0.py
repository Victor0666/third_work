import numpy as np
RULE_METADATA = {'structure_hash': 'b137ab1cdf2836d546d51a38dbe3127b2b7fb57ef6aa479c019e6d364d1154b1', 'parameter_schema_hash': '5ec2cb9f1981e760448418185ec6dc67760e0587bf56c351a19ecb4d82a9f117', 'best_parameter_hash': 'b06d125eba8e9cf99e7c5be916b47779e022b931f5379fdae6c95846fcfad0ba', 'best_parameters': {'epsilon': 0.002160180545699678, 'energy_duration_ratio_weight': 1.1532174046962025, 'uncertainty_dispersion_scale': 0.7908266261936184, 'urgency_cap_exponent': 0.5704120631362741, 'bottleneck_uncertainty_amplification': 0.7372456078017002, 'wait_saturation_scale': 3.0090271763132255, 'ddl_risk_gate_threshold': 0.6931010848614138, 'critical_path_coupling': 0.3516076861198866}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': '47ffa33ae38c69b58a0d1a87efe327998fafdcbb3492bd5a76452f4a8d10aac6', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining strengths of both parents:
      - Dominant pre-normalized neg_slack preserves hard-deadline fidelity.
      - Adaptive normalization using uncertainty dispersion for risk-contextual scaling.
      - Conditional DDL risk gating activates bottleneck & critical-path terms only when slack is critically low.
      - Bounded sigmoid wait saturation ensures robust anti-starvation.
      - Critical-path coupling gated by DDL risk avoids over-prioritizing non-critical paths when deadlines are safe.
      - All operations finite, deterministic, and use only {-2,-1,0,1,2} literals.
    """
    eps = 0.002160180545699678
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
        dispersion = 0.7908266261936184 * (unc_std + eps)
        fallback_range = np.max(x) - np.min(x) if N > 0 else eps
        denom = dispersion if dispersion > eps else fallback_range
        return (x - center) / (denom + eps)
    neg_slack = np.clip(-slack, 0.0, None)
    median_slack = np.median(slack) if N > 0 else 0.0
    ddl_risk_mask = (slack < 0.6931010848614138 * (median_slack + eps)).astype(float)
    urgency_linear = np.clip(median_slack - slack, 0.0, 2.0)
    non_neg_slack = np.clip(slack, 0.0, None)
    max_non_neg_slack = np.max(non_neg_slack) if N > 0 else eps
    urgency_cap = np.power(max_non_neg_slack + eps, 0.5704120631362741)
    urgency_linear = np.minimum(urgency_linear, urgency_cap)
    norm_urgency = adaptive_normalize(urgency_linear)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy_eff = adaptive_normalize(energy_per_duration)
    bottleneck_pressure = duration * upward_rank * remaining_work * (1.0 + urgency_linear + eps)
    unc_normalized = adaptive_normalize(uncertainty)
    bottleneck_pressure = bottleneck_pressure * np.power(1.0 + unc_normalized, 0.7372456078017002)
    norm_bottleneck = adaptive_normalize(bottleneck_pressure) * ddl_risk_mask
    critical_pressure = upward_rank * remaining_work
    critical_pressure = critical_pressure * (1.0 + ddl_risk_mask * (0.3516076861198866 - 1.0))
    norm_critical_pressure = adaptive_normalize(critical_pressure)
    wait_scaled = ready_wait_time / (3.0090271763132255 + eps)
    wait_clipped = np.clip(wait_scaled, -2.0, 2.0)
    wait_saturation = 1.0 / (1.0 + np.exp(-wait_clipped))
    norm_wait = adaptive_normalize(wait_saturation)
    score = neg_slack + norm_urgency + norm_bottleneck + norm_critical_pressure + 1.1532174046962025 * norm_energy_eff - norm_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
