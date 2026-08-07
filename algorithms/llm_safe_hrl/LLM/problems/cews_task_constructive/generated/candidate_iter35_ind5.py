import numpy as np
RULE_METADATA = {'structure_hash': '8746a50737efb033dc9682b7d720442da7e08af566f27b8e65ef79edba0b002b', 'parameter_schema_hash': 'e3c86dfbffd7b2edb7868106b626b9793b321238bff6e11d40125e0e9e465fb8', 'best_parameter_hash': '5c19a508905c1bf1f68eb01718ba9e6cd85ee0fd2e7702a8f26511660cd91772', 'best_parameters': {'epsilon': 1.0029748806478046e-06, 'ddl_risk_gate_slack_ratio': 0.29045521042828326, 'ddl_risk_gate_uncertainty_threshold': 0.19111445584299158, 'upward_rank_remaining_work_weight': 0.36309720747100027, 'successor_release_penalty_weight': 1.8468768640690925, 'uncertainty_sensitivity_weight': 1.6307043878160756, 'energy_uncertainty_coupling': 0.6440504008613187, 'wait_saturation_exponent': 1.1465124567651606, 'slack_feasibility_exponent': 0.7307501505258318}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': '1f0c03b275478a39c149a8f88f9e0830958afb2c546664644562e485408b4e9c', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining best structural elements from both parents:
      - Joint DDL-risk gate: activated when (slack <= median_slack * ratio) AND (normalized_uncertainty > threshold)
      - Uncertainty-scaled successor-release penalty: penalizes tasks whose successors have high remaining_work but low slack, amplified by uncertainty under joint risk.
      - Critical-path pressure: upward_rank × remaining_work, gated only under joint DDL+uncertainty risk.
      - Uncertainty-modulated energy penalty: marginal energy scaled by (1 + coupling * uncertainty), preserving direct physical interpretation.
      - Bounded power-law wait saturation (not decay): monotonic fairness boost with robust clipping and MAD-free normalization.
      - All normalizations use median/ptp with epsilon fallbacks; no std-based dispersion measures to avoid NaN on degenerate inputs.
      - Final score preserves hard-deadline dominance: neg_slack >> gated penalties >> energy >> wait_saturation.
      - Exactly one conditional branch (DDL gate); all operations vectorized and finite.
    """
    eps = 1.0029748806478046e-06
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
    high_risk_slack_condition = slack <= median_slack * 0.29045521042828326 + eps
    unc_min = np.min(uncertainty) if N > 0 else 0.0
    unc_max = np.max(uncertainty) if N > 0 else eps
    normalized_uncertainty = (uncertainty - unc_min) / (unc_max - unc_min + eps)
    high_uncertainty_condition = normalized_uncertainty > 0.19111445584299158
    ddl_risk_gate = np.where(high_risk_slack_condition & high_uncertainty_condition, 1.0, 0.0)
    critical_path_pressure = upward_rank * remaining_work
    gated_critical_pressure = ddl_risk_gate * 0.36309720747100027 * critical_path_pressure
    slack_feasibility = np.clip(slack / (np.abs(median_slack) + eps), 0.0, 1.0)
    slack_feasibility_sharpened = np.power(slack_feasibility + eps, 0.7307501505258318)
    successor_release_penalty = remaining_work * (1.0 - slack_feasibility_sharpened)
    gated_successor_penalty = ddl_risk_gate * successor_release_penalty * 1.8468768640690925 * (1.0 + 1.6307043878160756 * normalized_uncertainty)
    energy_penalty = min_incremental_energy * (1.0 + 0.6440504008613187 * uncertainty)
    wait_base = np.maximum(np.mean(ready_wait_time), eps) if N > 0 else eps
    wait_scaled = np.clip(ready_wait_time / (wait_base + eps), 0.0, 2.0)
    wait_saturation = np.power(wait_scaled + eps, 1.1465124567651606)
    score = neg_slack + gated_successor_penalty + gated_critical_pressure + energy_penalty - wait_saturation
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
