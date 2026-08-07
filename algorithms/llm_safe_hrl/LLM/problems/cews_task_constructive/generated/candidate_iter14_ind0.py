import numpy as np
RULE_METADATA = {'structure_hash': 'f1778881d566d77b9b821c42ecadd0bce1b9151f443490f12e442beff83c5316', 'parameter_schema_hash': 'dee749feccc1b03e753778d3ff213179ebda6085c863d9a0ebe6e54f5cec09f2', 'best_parameter_hash': '65b795755410cd6b2a4f8ef6010a7dd150b7706b0ff1f6f1b2b27b60d90fd045', 'best_parameters': {'epsilon': 0.00023661446695446919, 'ddl_risk_quantile_threshold': 0.6972519816803201, 'successor_bottleneck_coupling': 0.01031969873592355, 'iqr_low_percentile': 27.131251443184116, 'iqr_high_percentile': 86.17275443130745, 'wait_integration_strength': 0.0996378962479818, 'urgency_std_scale': 2.991911119539259}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': '1238332e1aee9b271a24f1102315b145800fa70b7118ffdf1fdd0915894406fd', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining Parent 2's unified bottleneck and Parent 1's robust normalization,
    enhanced with performance-analysis insights:
    - Replaces tanh urgency with stable piecewise-linear urgency: clipped (median_slack - slack) / (std + eps)
    - Uses percentile-based DDL-risk gate: (slack < 0) & (uncertainty > quantile(uncertainty, ddl_risk_quantile_threshold))
    - Integrates ready_wait_time directly into bottleneck (not subtracted), scaled only under DDL risk
    - Removes redundant parameters; keeps exactly 7 tunable scalars (5–12 range satisfied)
    - All numeric literals are in {-2,-1,0,1,2}; no hidden constants.
    - Adaptive normalization uses IQR only (no fallback) — tuned percentiles ensure stability across workloads.
    """
    eps = 0.00023661446695446919
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
        q_low = np.percentile(x, 27.131251443184116)
        q_high = np.percentile(x, 86.17275443130745)
        iqr = q_high - q_low + eps
        center = np.median(x)
        return (x - center) / iqr
    median_slack = np.median(slack) if N > 0 else 0.0
    std_slack = np.std(slack) if N > 1 else eps
    urgency_linear = np.clip(median_slack - slack, 0.0, np.inf)
    urgency = np.clip(urgency_linear / (std_slack * 2.991911119539259 + eps), 0.0, 1.0)
    norm_urgency = adaptive_normalize(urgency)
    duration = min_exec_time + min_comm_time
    quantile_uncertainty = np.quantile(uncertainty, 0.6972519816803201) if N > 0 else eps
    ddl_risk_gate = ((slack < 0.0) & (uncertainty > quantile_uncertainty)).astype(float)
    wait_contribution = ready_wait_time * ddl_risk_gate * 0.0996378962479818
    bottleneck_pressure = duration * upward_rank * (1.0 + urgency) * (1.0 + uncertainty) * (min_incremental_energy + eps) * (1.0 + wait_contribution)
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    score = norm_urgency + 0.01031969873592355 * norm_bottleneck
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
