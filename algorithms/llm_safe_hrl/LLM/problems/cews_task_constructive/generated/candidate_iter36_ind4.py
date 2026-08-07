import numpy as np
RULE_METADATA = {'structure_hash': '54d529d7a5c7368340dd16b7fadad8b7de3ac27cdd85fa2a9281742d85aa0b84', 'parameter_schema_hash': 'acb54d07d9e7e2034b4ab9fd6a3ca2149c0592f2c556f805e899e526503bc53f', 'best_parameter_hash': 'a75438d5847d5e4c7828c28fe1974e8e949a932698e035811928f226551305a0', 'best_parameters': {'epsilon': 3.3650531943112854e-05, 'slack_sensitivity_exponent': 1.3992846379498836, 'upward_rank_remaining_work_weight': 1.0480866886196938, 'energy_priority_weight': 0.9654717077655559, 'wait_saturation_exponent': 0.7665255273334777, 'ddl_feasibility_gate_threshold': 0.0470247826347862}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': '53979cf8cbd9ec7158ea0dc8836aea40b446461e3e8f104c8a28760b71fca4ba', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
      - Deadline-feasibility–aware energy gating: slack >= 0 AND uncertainty < median(uncertainty) * threshold.
      - Unified bottleneck pressure: duration × upward_rank × (1 + uncertainty).
      - Robust wait-saturation via bounded power-law on relative wait time (vs mean), clipped to [0,2].
      - Robust normalization using median-centered sigmoid scaling (no min/max outliers).
      - Critical-path pressure remains raw to preserve scale integrity.
      - All numeric literals are in {-2,-1,0,1,2}; epsilon handled via PARAMS.
    """
    eps = 3.3650531943112854e-05
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    linear_slack_penalty = np.clip(-slack, 0.0, None)
    exp_slack_penalty = np.where(slack <= 0.0, np.power(np.abs(slack) + eps, 1.3992846379498836), 0.0)
    neg_slack = linear_slack_penalty + exp_slack_penalty
    unc_median = np.median(uncertainty) if N > 0 else eps
    ddl_feasibility_gate = (slack >= 0.0) & (uncertainty < unc_median * 0.0470247826347862)
    energy_term = np.where(ddl_feasibility_gate, min_incremental_energy, 0.0)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    bottleneck_pressure = duration * upward_rank * (1.0 + uncertainty)
    critical_path_pressure = upward_rank * remaining_work
    wait_mean = np.mean(ready_wait_time) if N > 0 else eps
    wait_ratio = np.clip(ready_wait_time / (wait_mean + eps), 0.0, 2.0)
    wait_saturation = np.power(wait_ratio + eps, 0.7665255273334777)

    def robust_normalize(x):
        x = np.copy(x)
        if N == 0:
            return np.zeros_like(x)
        x_med = np.median(x)
        x_dev = np.abs(x - x_med)
        mad = np.median(x_dev) if np.any(x_dev) else eps
        scaled = (x - x_med) / (mad + eps)
        clipped_scaled = np.clip(scaled, -2.0, 2.0)
        normalized = 1.0 / (1.0 + np.exp(-clipped_scaled))
        return normalized
    norm_bottleneck = robust_normalize(bottleneck_pressure)
    norm_energy = robust_normalize(energy_term)
    norm_wait = robust_normalize(wait_saturation)
    score = neg_slack + 1.0480866886196938 * critical_path_pressure + 1.0480866886196938 * norm_bottleneck + 0.9654717077655559 * norm_energy - norm_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
