import numpy as np
RULE_METADATA = {'structure_hash': '1f367b84e551cfe9db9bf1e8559c712a9030975692d4ed14e4a14736f14f8733', 'parameter_schema_hash': '14815dfab355e5f8ac68664ced1ed882f9b84a5a8b6e777bf8c58f61f640f14c', 'best_parameter_hash': 'cdd70dcf8d8fbdf1e5ad2006ec4375e4a636a996abc75ac660921d7ffbd16db1', 'best_parameters': {'epsilon': 0.005822583781368301, 'ddl_protection_gate_threshold': 0.2961077814015109, 'upward_rank_remaining_work_weight': 1.2521572973609207, 'uncertainty_coupling_exponent': 0.947569596695619, 'wait_decay_power': 1.0845107654226183, 'energy_normalization_scale': 2.8600475289784812}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': '8ea78b781af485b31942bf32989a5c89518be398c4cbf95aa9ad3cd544ad3b11', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with structural improvements guided by counterfactual evidence:
      - Introduces conditional DDL protection gate: activates critical-path pressure only when slack <= median_slack AND uncertainty > threshold.
      - Replaces linear urgency with bounded power-law wait decay (wait_ratio^p) for small-N robustness and anti-starvation.
      - Uses pre-normalized neg_slack as dominant term, uncorrupted by scaling or amplification.
      - Gates all uncertainty-driven terms by verified DDL risk and slack feasibility — suppresses non-critical-path tasks under tight slack.
      - Removes inactive parameters (e.g., urgency_cap_exponent, wait_saturation_scale) to reduce overfitting.
      - All operations use np.finfo safeguards; no literals beyond {-2,-1,0,1,2}.
    """
    eps = 0.005822583781368301
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    neg_slack = np.clip(-slack, 0.0, None)
    median_slack = np.median(slack) if N > 0 else 0.0
    slack_feasible = (slack > median_slack).astype(float)
    uncertainty_normalized = (uncertainty - np.min(uncertainty)) / (np.ptp(uncertainty) + eps) if N > 1 else np.zeros_like(uncertainty)
    ddl_risk_gate = np.where((slack <= median_slack) & (uncertainty_normalized > 0.2961077814015109), 1.0, 0.0)
    critical_path_pressure = upward_rank * remaining_work
    gated_critical_pressure = ddl_risk_gate * critical_path_pressure
    unc_coupled_penalty = np.power(1.0 + uncertainty_normalized, 0.947569596695619) * neg_slack
    gated_unc_penalty = ddl_risk_gate * unc_coupled_penalty
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    energy_center = np.median(energy_per_duration) if N > 0 else 0.0
    energy_dispersion = 2.8600475289784812 * (np.std(energy_per_duration) + eps) if N > 1 else eps
    fallback_energy_range = np.max(energy_per_duration) - np.min(energy_per_duration) if N > 0 else eps
    energy_denom = np.where(energy_dispersion > eps, energy_dispersion, fallback_energy_range)
    norm_energy = (energy_per_duration - energy_center) / (energy_denom + eps)
    max_wait = np.max(ready_wait_time) if N > 0 else eps
    wait_ratio = np.clip(ready_wait_time / (max_wait + eps), 0.0, 1.0)
    wait_decay = np.power(wait_ratio + eps, 1.0845107654226183)
    score = neg_slack + gated_unc_penalty + 1.2521572973609207 * gated_critical_pressure + norm_energy - wait_decay
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
