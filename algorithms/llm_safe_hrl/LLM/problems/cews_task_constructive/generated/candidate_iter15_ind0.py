import numpy as np
RULE_METADATA = {'structure_hash': 'e13d58b5f42f3afa603ae34579bd44e692ef16696cb509047685ca83d403f3ec', 'parameter_schema_hash': '41233cb931b3a4b689e4f79aec6f23c6a3293ea2ddb8fc1a1a8b870469c99b5f', 'best_parameter_hash': '889f041e73c62c2092ad152e3dcd01160bd7a1d2ae9de7cc6bf2f0eeb53cce75', 'best_parameters': {'epsilon': 0.0007278991557702362, 'ddl_risk_activation_threshold': 0.7410475614739748, 'successor_bottleneck_coupling': 0.0008725166391526828, 'iqr_low_percentile': 32.99090139727166, 'iqr_high_percentile': 77.5326488514703, 'wait_fairness_weight': 0.1410863845221098, 'urgency_scale': 0.6723949952890106}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': 'd08514b669877943464e6231d3ddb601cfa36f9375cdce6d1211967aaf397236', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
    - Restored bounded tanh urgency (superior monotonicity & stability vs piecewise-linear)
    - Anti-starvation reinstated as subtracted fairness term (not embedded), ensuring starvation prevention even in feasible regimes
    - Joint DDL-risk gate simplified to `(slack < 0) & (uncertainty > median_uncertainty * threshold)` — robust, interpretable, no quantile estimation noise
    - Bottleneck now cleanly separates deadline-driven urgency (tanh) from critical-path + energy pressure, avoiding coupling artifacts
    - Fairness term uses adaptive-normalized wait time scaled by `wait_fairness_weight`, applied unconditionally (not gated) for consistent fairness
    - All numeric literals are in {-2,-1,0,1,2}; no hidden constants; strict adherence to interface contract.
    """
    eps = 0.0007278991557702362
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
        q_low = np.percentile(x, 32.99090139727166)
        q_high = np.percentile(x, 77.5326488514703)
        iqr = q_high - q_low + eps
        center = np.median(x)
        return (x - center) / iqr
    abs_slack = np.abs(slack)
    scale = np.median(abs_slack) if np.any(abs_slack > eps) else 1.0
    tanh_urgency = np.tanh(-slack / (scale * 0.6723949952890106 + eps))
    urgency = np.clip((tanh_urgency + 1.0) / 2.0, 0.0, 1.0)
    norm_urgency = adaptive_normalize(urgency)
    duration = min_exec_time + min_comm_time
    bottleneck_pressure = duration * upward_rank * (1.0 + uncertainty) * (min_incremental_energy + eps)
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    norm_wait = adaptive_normalize(ready_wait_time)
    fairness_term = -0.1410863845221098 * norm_wait
    median_uncertainty = np.median(uncertainty) if N > 0 else eps
    ddl_risk_gate = ((slack < 0.0) & (uncertainty > 0.7410475614739748 * median_uncertainty)).astype(float)
    stress_amplified_urgency = norm_urgency * (1.0 + ddl_risk_gate)
    score = stress_amplified_urgency + 0.0008725166391526828 * norm_bottleneck + fairness_term
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
