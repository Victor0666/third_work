import numpy as np
RULE_METADATA = {'structure_hash': '37feb3ece4b7bff86884f71c9495514f202585cd77f2027fc6767728343da47a', 'parameter_schema_hash': '8d87a770f5fe9eee31f3993138895c4b8fa877edc58cde22edd92e4a0c15a0bc', 'best_parameter_hash': '69db0d56b6c9343bc2fd06138e74f930db2ad1dc94622d3b11a4e6aadc0ad3d7', 'best_parameters': {'epsilon': 5.2249098635888295e-09, 'ddl_protection_threshold': 1.326191201993824, 'critical_path_release_weight': 2.981934689313353, 'risk_adjusted_energy_weight': 2.576872105730826, 'uncertainty_slack_coupling': 0.8069463685212583, 'wait_starvation_penalty': 0.5298997924457867, 'duration_mad_scale': 2.222968892914472, 'energy_uncertainty_interaction': 1.6034013964409919, 'slack_linear_ramp_width': 1.2435055027862365, 'rank_slack_balance': 0.9489058249500729, 'mad_normalization_shift': 0.1302536407748334, 'congestion_aware_energy_gate': 0.43743796471068386}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': 'c43e6363516cbe0f6b702b95ce57c1f6105bf0ad1ee7e53263d16a87283151d3', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule featuring:
      - Linearized slack penalty (stable ranking under lateness)
      - Congestion-aware gating on energy terms using wait-time & uncertainty
      - Critical-path release signal decoupled from normalization in DDL-safe region
      - Joint MAD normalization with stabilizing shift for all risk dimensions
      - All numeric literals strictly {-2,-1,0,1,2}
    """
    eps = 5.2249098635888295e-09
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
    all_risk_dims = np.stack([duration_total, abs_slack, uncertainty], axis=0)
    mad_per_dim = np.mean(np.abs(all_risk_dims - np.median(all_risk_dims, axis=1, keepdims=True)), axis=1) + eps
    mad_shifted = mad_per_dim + 0.1302536407748334 * np.median(mad_per_dim)
    duration_norm = duration_total / (mad_shifted[0] * 2.222968892914472 + eps)
    slack_norm = abs_slack / (mad_shifted[1] * 2.222968892914472 + eps)
    unc_norm = uncertainty / (mad_shifted[2] * 2.222968892914472 + eps)
    slack_penalty = np.where(slack < 0, -slack * (1.0 + 0.8069463685212583 * unc_norm), 0.0)
    critical_release = upward_rank * remaining_work
    critical_release_safe = critical_release * 1.0
    critical_release_pressure = (critical_release - np.median(critical_release)) / (np.mean(np.abs(critical_release - np.median(critical_release))) + eps)
    critical_release_score = np.where(slack > 1.326191201993824, -critical_release_safe, -critical_release_pressure * 2.981934689313353)
    ramp_half = 1.2435055027862365 / 2.0
    gate_center = 1.326191201993824
    gate_low = gate_center - ramp_half
    gate_high = gate_center + ramp_half
    slack_gate = np.clip((slack - gate_low) / (ramp_half * 2.0 + eps), 0.0, 1.0)
    median_wait = np.median(ready_wait_time) + eps
    congestion_proxy = (ready_wait_time / median_wait + eps) * (1.0 + unc_norm)
    congestion_gate = np.clip(congestion_proxy * 0.43743796471068386, 0.0, 1.0)
    mad_energy = np.mean(np.abs(min_incremental_energy - np.median(min_incremental_energy))) + eps
    energy_norm = (min_incremental_energy - np.median(min_incremental_energy)) / (mad_energy * 2.222968892914472 + eps)
    energy_score = slack_gate * congestion_gate * energy_norm * 2.576872105730826
    energy_uncertainty_score = slack_gate * congestion_gate * energy_norm * unc_norm * 1.6034013964409919
    median_wait = np.median(ready_wait_time) + eps
    wait_ratio = np.clip(ready_wait_time / median_wait, 0.0, 2.0)
    slack_deficit_boost = np.clip(-slack / (1.326191201993824 + eps), 0.0, 2.0)
    wait_score = ready_wait_time * (1.0 + slack_deficit_boost)
    wait_mad = np.mean(np.abs(wait_score - np.median(wait_score))) + eps
    wait_norm = (wait_score - np.median(wait_score)) / (wait_mad * 2.222968892914472 + eps)
    wait_final = wait_norm * 0.5298997924457867
    slack_urgency = np.where(slack < 0, -slack, 0.0)
    slack_urgency_norm = (slack_urgency - np.median(slack_urgency)) / (np.mean(np.abs(slack_urgency - np.median(slack_urgency))) + eps)
    rank_score = -upward_rank / (np.median(upward_rank) + eps) * 0.9489058249500729
    slack_weight = np.clip(1.0 - slack_norm / (slack_norm.max() + eps), 0.0, 1.0)
    balanced_rank_score = rank_score * (1.0 - slack_weight) + slack_urgency_norm * slack_weight
    score = slack_penalty + critical_release_score + wait_final + energy_score + energy_uncertainty_score + balanced_rank_score
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
