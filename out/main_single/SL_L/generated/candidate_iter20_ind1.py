import numpy as np
RULE_METADATA = {'structure_hash': 'fcb96474f7bbb4adede67dc8c1672381d512ec2d8c27e65f662d479ad266954a', 'parameter_schema_hash': '03bf96266d8643f2d9f32543e73aef5e00df21d2c5b01b2c9fc93c752c52ad22', 'best_parameter_hash': '4fe80c760b38e13705950e94db19a9fa374d1486fc4f46cc395315cc40340a3f', 'best_parameters': {'epsilon': 1.0791696486367572e-06, 'ddl_protection_threshold': 0.9533115490728367, 'critical_path_release_weight': 1.3725224972655465, 'risk_adjusted_energy_weight': 1.1336617177880608, 'uncertainty_slack_interaction': 0.8978580228374229, 'wait_starvation_penalty': 0.7549535846323343, 'duration_mad_scale': 0.9491300947799948, 'energy_uncertainty_coupling': 0.4882001668267748, 'ddl_violation_exponent': 1.499311659455103, 'critical_pressure_decay': 0.7026109282065933}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': 'd211d2c5e3de0e40435eb5014b974d2d882308655d918f95a73b97ee7d6515cb', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining robustness from Parent 2 and structural clarity from Parent 1:
      - Joint MAD normalization over |slack|, uncertainty, and duration_total (robust scaling)
      - Lexicographic DDL enforcement: violation penalty raised to power >1 for strict dominance
      - Critical-path urgency uses smooth pressure decay (not binary gating) to avoid discontinuities
      - Wait-term uses relative wait efficiency (ready_wait_time / duration_total) instead of absolute time
      - Energy-uncertainty coupling gated by bounded sigmoid on uncertainty (centered at 1.0)
      - All numeric literals restricted to {-2,-1,0,1,2}; no other constants
      - No percentile clipping or fragile medians on empty subsets; safe fallbacks used
    """
    eps = 1.0791696486367572e-06
    finfo = np.finfo(float)
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=finfo.max, neginf=finfo.min)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=finfo.max, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=eps, posinf=finfo.max, neginf=eps)
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)
    duration_total = min_exec_time + min_comm_time + eps
    abs_slack = np.abs(slack) + eps
    all_risk_features = np.stack([abs_slack, uncertainty, duration_total], axis=0)
    mad_per_dim = np.mean(np.abs(all_risk_features - np.median(all_risk_features, axis=1, keepdims=True)), axis=1) + eps
    joint_mad = np.median(mad_per_dim) + eps

    def robust_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        centered = x - np.median(x)
        denom = joint_mad * 0.9491300947799948 + eps
        return np.clip(centered / denom, -2.0, 2.0)
    ddl_violation_base = np.maximum(0.0, 0.9533115490728367 - slack)
    ddl_violation_penalty = np.power(ddl_violation_base + eps, 1.499311659455103)
    critical_release_score = upward_rank * remaining_work
    slack_ratio = np.clip((0.9533115490728367 - slack) / (0.9533115490728367 + eps), 0.0, 1.0)
    critical_weight = 1.3725224972655465 * (0.7026109282065933 + (1.0 - 0.7026109282065933) * slack_ratio)
    critical_score = -robust_normalize(critical_release_score) * critical_weight
    feasible_mask = np.where(slack > 0.9533115490728367, 1.0, 0.0)
    energy_norm = robust_normalize(min_incremental_energy)
    energy_term = feasible_mask * 1.1336617177880608 * energy_norm
    unc_slack_interaction = uncertainty * (0.9533115490728367 - slack) * 0.8978580228374229
    unc_slack_norm = robust_normalize(unc_slack_interaction)
    wait_efficiency = np.where(duration_total > eps, ready_wait_time / duration_total, 0.0)
    wait_norm = robust_normalize(wait_efficiency)
    wait_term = feasible_mask * 0.7549535846323343 * wait_norm
    unc_sigmoid = 1.0 / (1.0 + np.exp(-2.0 * (1.0 - uncertainty)))
    energy_uncertainty_term = feasible_mask * 0.4882001668267748 * energy_norm * unc_sigmoid
    score = robust_normalize(ddl_violation_penalty) + critical_score + unc_slack_norm
    score += energy_term + wait_term + energy_uncertainty_term
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
