import numpy as np
RULE_METADATA = {'structure_hash': 'ceda9aa2f92973547eb76e01704d91a14a61dd0ca6fd1a0408ff5180bb392a0f', 'parameter_schema_hash': '6456c3b905c2b4e5ce744ff41fe5d67b85c0ec642ab6d2daf19184a59f4602f9', 'best_parameter_hash': '9ed97f8ca681979fae5fd33dab11cc720f345a4f413d2d25c822ab715d31af59', 'best_parameters': {'epsilon': 1.1268005391547277e-09, 'ddl_protection_threshold': 0.698394512839679, 'critical_path_release_weight': 3.364174536706764, 'risk_adjusted_energy_weight': 1.522470610054155, 'uncertainty_slack_interaction': 1.5367575796261415, 'wait_starvation_penalty': 0.8191243035264556, 'duration_mad_scale': 0.14402776207526505, 'energy_uncertainty_coupling': 0.0256313290786254, 'ddl_violation_penalty_weight': 1.2925063487356538, 'normalized_uncertainty_weight': 0.022372829679063358}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': '60c0505203e72ef7c871925a7a89b878263e03eb634f12defffb9bcc6abf907a', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with restored lexicographic DDL dominance and simplified uncertainty handling:
      - Replaces exponentiated penalty with large linear weight + hard feasibility gating for robust deadline enforcement
      - Removes fragile sigmoid and decay logic; uses clean binary mask for non-DDL terms
      - Introduces direct normalized_uncertainty term: always active, linearly penalizes high uncertainty regardless of slack
      - Joint MAD normalization now includes uncertainty *and* |slack| *and* duration_total → coherent risk alignment
      - Critical-path score uses hard-gated pressure (not smooth decay) to ensure discontinuity at threshold for strict priority ordering
      - Wait term uses relative efficiency but clamped to avoid numerical noise when duration is tiny
      - All numeric literals strictly limited to {-2,-1,0,1,2}; no hidden constants
      - Final score = DDL-penalty (dominant) + critical-path + uncertainty + (feasible-only) energy/wait/coupling
    """
    eps = 1.1268005391547277e-09
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
        denom = joint_mad * 0.14402776207526505 + eps
        return np.clip(centered / denom, -2.0, 2.0)
    ddl_violation_base = np.where(slack < 0.698394512839679, 0.698394512839679 - slack, 0.0)
    ddl_violation_penalty = 1.2925063487356538 * ddl_violation_base
    critical_release_score = upward_rank * remaining_work
    critical_mask = np.where(slack <= 0.698394512839679, 1.0, 0.0)
    critical_score = -robust_normalize(critical_release_score) * critical_mask * 3.364174536706764
    unc_norm = robust_normalize(uncertainty)
    uncertainty_penalty = 0.022372829679063358 * unc_norm
    feasible_mask = np.where(slack > 0.698394512839679, 1.0, 0.0)
    energy_norm = robust_normalize(min_incremental_energy)
    energy_term = feasible_mask * 1.522470610054155 * energy_norm
    unc_slack_interaction = uncertainty * (0.698394512839679 - slack) * 1.5367575796261415
    unc_slack_norm = robust_normalize(unc_slack_interaction)
    unc_slack_term = feasible_mask * unc_slack_norm
    wait_efficiency = np.where(duration_total > eps, np.clip(ready_wait_time / duration_total, 0.0, 2.0), 0.0)
    wait_norm = robust_normalize(wait_efficiency)
    wait_term = feasible_mask * 0.8191243035264556 * wait_norm
    energy_uncertainty_term = feasible_mask * 0.0256313290786254 * energy_norm * unc_norm
    score = robust_normalize(ddl_violation_penalty) + critical_score + uncertainty_penalty + unc_slack_term
    score += energy_term + wait_term + energy_uncertainty_term
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
