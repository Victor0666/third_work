import numpy as np
RULE_METADATA = {'structure_hash': 'f7d6dbe7b2dd3e3218bcd50bf68a7bf7bb0de5b785644ab2d17d678bb1c450d9', 'parameter_schema_hash': '83c6f0ecd7e50edb9ba6c4c90c57ccd4ce95a32e156ee058baeb6b7661802c12', 'best_parameter_hash': 'a9be1021fc75162b8c9147e914009a30d1eca67115891926d6ce24879cda0fbe', 'best_parameters': {'epsilon': 5.919818566096707e-09, 'ddl_protection_threshold': 1.4901782873900427, 'critical_path_release_weight': 3.2233081098277365, 'risk_adjusted_energy_weight': 0.7099706391283505, 'uncertainty_slack_coupling': 1.2709908562295407, 'wait_starvation_penalty': 0.2522051014689666, 'duration_mad_scale': 1.2147880493407675, 'energy_uncertainty_interaction': 0.15625386479406245, 'slack_sigmoid_steepness': 4.19721301861527, 'rank_slack_balance': 0.8649670515868039, 'duration_risk_exponent': 1.1863716778973126, 'energy_mad_adaptation': 0.3666898767108344}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': 'd0fb5ad2a23b1032ac5291bfca73b004b6940ef7edd323a17419a1c5c5cad6e9', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining Parent 2's robustness with novel adaptive normalization and risk-exponentiated duration penalty:
      - Retains smooth sigmoid gating on slack (Parent 2) for fuzzy feasibility stability
      - Introduces duration_risk_exponent to nonlinearly amplify high-uncertainty-duration risk (novel structural change)
      - Replaces fixed MAD scaling with energy_mad_adaptation that scales normalization by median energy magnitude → improves cross-scenario transfer
      - Keeps joint MAD normalization over [duration_total, |slack|, uncertainty] for coherent risk alignment (validated)
      - Preserves starvation mitigation via urgency-weighted ready_wait_time and critical-path release signal
      - Uses bounded interpolation (rank_slack_balance) to avoid overreaction to extreme slack values
      - All tunables exposed; no literals beyond {-2,-1,0,1,2}; deterministic and finite-output guaranteed
    """
    eps = 5.919818566096707e-09
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
    duration_norm = duration_total / (mad_per_dim[0] * 1.2147880493407675 + eps)
    slack_norm = abs_slack / (mad_per_dim[1] * 1.2147880493407675 + eps)
    unc_norm = uncertainty / (mad_per_dim[2] * 1.2147880493407675 + eps)
    duration_risk_base = (min_exec_time + min_comm_time) * uncertainty ** 1.1863716778973126
    slack_penalty = np.where(slack < 0, -slack * (1.0 + 1.2709908562295407 * unc_norm) + duration_risk_base, duration_risk_base)
    critical_release = upward_rank * remaining_work
    critical_release_median = np.median(critical_release)
    critical_release_mad = np.mean(np.abs(critical_release - critical_release_median)) + eps
    critical_release_norm = (critical_release - critical_release_median) / critical_release_mad
    critical_release_score = -critical_release_norm * 3.2233081098277365
    slack_gate = 1.0 / (1.0 + np.exp(-4.19721301861527 * (slack - 1.4901782873900427)))
    energy_median = np.median(min_incremental_energy)
    energy_mad = np.mean(np.abs(min_incremental_energy - energy_median)) + eps
    energy_norm_scale = energy_median * 0.3666898767108344 + eps
    energy_norm = (min_incremental_energy - energy_median) / (energy_mad * energy_norm_scale + eps)
    energy_score = slack_gate * energy_norm * 0.7099706391283505
    energy_uncertainty_score = slack_gate * energy_norm * unc_norm * 0.15625386479406245
    slack_distance = np.clip(1.4901782873900427 - slack, 0.0, np.inf)
    wait_score = ready_wait_time * (1.0 + slack_distance / (1.4901782873900427 + eps))
    wait_median = np.median(wait_score)
    wait_mad = np.mean(np.abs(wait_score - wait_median)) + eps
    wait_norm = (wait_score - wait_median) / wait_mad
    wait_final = wait_norm * 0.2522051014689666
    slack_weight = np.clip(1.0 - slack_norm / (slack_norm.max() + eps), 0.0, 1.0)
    rank_weight = 1.0 - slack_weight
    rank_score = -upward_rank / (np.median(upward_rank) + eps) * rank_weight * 0.8649670515868039
    score = slack_penalty + critical_release_score + wait_final
    score += energy_score + energy_uncertainty_score + rank_score
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
