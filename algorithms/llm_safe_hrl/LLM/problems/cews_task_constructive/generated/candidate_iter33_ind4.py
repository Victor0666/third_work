import numpy as np
RULE_METADATA = {'structure_hash': '40549ea8a8ca2edda5a86fd90f98f93766b97de46fd4611bde75cde86a092af5', 'parameter_schema_hash': 'a8745f85060541ee8bfeb4e259912b41e8aecd32e0cba04834c65a9e082e2d9e', 'best_parameter_hash': 'c3e0437fcfd088d24b5adec8792f9ee415de1217c474393afc83153e0116492b', 'best_parameters': {'epsilon': 2.3151384509560893e-06, 'energy_duration_ratio_weight': 1.5263530838671504, 'uncertainty_dispersion_scale': 0.3881960869195954, 'urgency_cap_exponent': 0.7646274283563534, 'bottleneck_uncertainty_amplification': 1.2535035775640098, 'neg_slack_amplification': 0.5022258367260514, 'critical_path_slack_modulation': 0.33402687270808984, 'wait_fairness_strength': 0.6601424064739638, 'energy_uncertainty_coupling': 0.7896301122533064, 'uncertainty_sigmoid_steepness': 5.055421171335625, 'uncertainty_activation_threshold': 0.6271607612138697, 'ddl_protection_threshold': -0.4295480187861962}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': '826f83cfac65735b26dc90d3c0325af37325399f835071d2c0a5a2688d859932', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with two key structural improvements:
      - Bounded DDL protection gate: activates only when slack < ddl_protection_threshold,
        applying multiplicative boost to amplified_neg_slack to avoid over-penalization
        of marginally risky tasks while preserving hard-deadline dominance.
      - Unified adaptive normalization: replaces rank_normalize with uncertainty-scaled
        adaptive_normalize for all terms including energy_uncertainty_term, ensuring
        consistent dispersion semantics and improved robustness to skewed uncertainty.
      - Successor-release coupling approximated via task's own slack (conservative fallback),
        using exp(-λ × max(0, -slack)) — avoids need for external successor data while
        capturing downstream deadline pressure.
    """
    eps = 2.3151384509560893e-06
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    finfo = np.finfo(float)

    def adaptive_normalize(x):
        x = np.copy(x)
        center = np.median(x)
        unc_std = np.std(uncertainty) if N > 1 else eps
        dispersion = 0.3881960869195954 * (unc_std + eps)
        fallback_range = np.max(x) - np.min(x) if N > 0 else eps
        denom = dispersion if dispersion > eps else fallback_range
        return (x - center) / (denom + eps)
    neg_slack = np.clip(-slack, 0.0, None)
    amplified_neg_slack = 0.5022258367260514 * neg_slack
    ddl_gate = np.where(slack < -0.4295480187861962, 1.0, 0.0)
    gated_neg_slack = amplified_neg_slack * (1.0 + ddl_gate)
    median_slack = np.median(slack) if N > 0 else 0.0
    urgency_linear = np.clip(median_slack - slack, 0.0, 2.0)
    non_neg_slack = np.clip(slack, 0.0, None)
    max_non_neg_slack = np.max(non_neg_slack) if N > 0 else eps
    urgency_cap = np.power(max_non_neg_slack + eps, 0.7646274283563534)
    urgency_linear = np.minimum(urgency_linear, urgency_cap)
    norm_urgency = adaptive_normalize(urgency_linear)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy_eff = adaptive_normalize(energy_per_duration)
    bottleneck_pressure = duration * upward_rank * remaining_work * (1.0 + urgency_linear + eps)
    unc_normalized = adaptive_normalize(uncertainty)
    bottleneck_pressure = bottleneck_pressure * np.power(1.0 + unc_normalized, 1.2535035775640098)
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    successor_release_factor = np.exp(--0.4295480187861962 * neg_slack)
    critical_pressure = upward_rank * remaining_work * np.power(1.0 + neg_slack + eps, 0.33402687270808984) * successor_release_factor
    norm_critical_pressure = adaptive_normalize(critical_pressure)
    unc_centered = uncertainty - 0.6271607612138697
    unc_clipped = np.clip(unc_centered, -2.0, 2.0)
    unc_gate = 1.0 / (1.0 + np.exp(-5.055421171335625 * unc_clipped))
    energy_uncertainty_term = min_incremental_energy * unc_gate
    norm_energy_uncertain = adaptive_normalize(energy_uncertainty_term)
    norm_wait = adaptive_normalize(ready_wait_time)
    score = gated_neg_slack + norm_urgency + norm_bottleneck + norm_critical_pressure + 1.5263530838671504 * norm_energy_eff + 0.7896301122533064 * norm_energy_uncertain - 0.6601424064739638 * norm_wait
    score = np.nan_to_num(score, nan=finfo.max, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
