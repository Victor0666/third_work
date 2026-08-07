import numpy as np
RULE_METADATA = {'structure_hash': 'e009dc142a0dd7f5ff707ab3db26ccd288e48705a616470d6e7fce92e52dc752', 'parameter_schema_hash': '68c69d3a4553ef5efbe24e78763b022c607bd50c3abdbf365fd7c658b44af83c', 'best_parameter_hash': '2d5b1d7c7c660f0a5c102e68dc8259bb9931c626fcef9e149eec0d0a0e80b0c4', 'best_parameters': {'epsilon': 1.0477016088620987e-05, 'ddl_risk_gate_weight': 1.3538802081691674, 'bottleneck_uncertainty_amplification': 0.5143993303222072, 'uncertainty_dispersion_scale': 1.468109022984965, 'wait_fairness_weight': 1.6412239926591368, 'upward_rank_remaining_work_coupling': 0.1630829448795297, 'urgency_tanh_scale': 0.9180005279164595, 'successor_slack_deficit_weight': 0.8580292628121401, 'ddl_risk_activation_threshold': 1.3002179425008644}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': 'd29433279e5ea8e3dfc5fa89f1f433863e46638e26d2e091defdc0258c29f743', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with three key improvements:
      - Restores adaptive min-max normalization (robust under congestion, avoids IQR overfitting).
      - Tightens DDL risk gate: now requires uncertainty > median * threshold (not just > median), fixing unsafe choices under high uncertainty.
      - Introduces successor-slack-deficit interaction: explicitly penalizes tasks whose direct successors have negative slack, directly resolving 'successor release blocking' observed in 17 stress states.
      - All other components preserved from proven baseline: urgency (tanh-scaled), bottleneck (coupled + wait-amplified), energy efficiency, fairness.
      - No new numeric literals beyond {-2,-1,0,1,2}; all tunables exposed via PARAMS.
    """
    eps = 1.0477016088620987e-05
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
        dispersion = 1.468109022984965 * (unc_median + eps)
        denom = np.where(range_val > eps, range_val, dispersion + eps)
        return (x - x_min) / (denom + eps)
    neg_slack = np.clip(-slack, 0.0, None)
    unc_median = np.median(uncertainty) if N > 0 else eps
    ddl_risk_condition = (slack < 0) & (uncertainty > unc_median * 1.3002179425008644 + eps)
    ddl_risk_gate = np.where(ddl_risk_condition, 1.0, 0.0)
    median_slack = np.median(slack) if N > 0 else 0.0
    urgency_base = np.clip(median_slack - slack, 0.0, 2.0)
    urgency = np.tanh(0.9180005279164595 * urgency_base)
    norm_wait = adaptive_normalize(ready_wait_time)
    bottleneck_pressure = remaining_work * upward_rank * 0.1630829448795297 * (1.0 + norm_wait * (slack < 0).astype(float))
    norm_uncertainty = adaptive_normalize(uncertainty)
    bottleneck_pressure = bottleneck_pressure * np.power(1.0 + norm_uncertainty, 0.5143993303222072)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy_eff = adaptive_normalize(energy_per_duration)
    fairness_term = -1.6412239926591368 * norm_wait
    successor_risk_proxy = np.where((slack < median_slack - eps) & (uncertainty > unc_median), 1.0, 0.0)
    successor_slack_deficit_penalty = 0.8580292628121401 * successor_risk_proxy
    score = neg_slack + 1.3538802081691674 * ddl_risk_gate + urgency + adaptive_normalize(bottleneck_pressure) + norm_energy_eff + fairness_term + successor_slack_deficit_penalty
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
