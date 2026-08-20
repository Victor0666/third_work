import numpy as np
RULE_METADATA = {'structure_hash': '4aeee8e73448dc10d652af376d45ec0859c54b040de7958b163be1daac6d954a', 'parameter_schema_hash': 'd05b1086cf35f301af0b7490a3e74c153d952d286f9b635b18c89fa9eda0cfc1', 'best_parameter_hash': '749a013383ef0c1f5ef2b345116249520e3c16eba27a825cac1b1f8a16f5c2a2', 'best_parameters': {'epsilon': 5.3385149082474506e-05, 'slack_penalty_exponent': 2.996226597663675, 'criticality_scale': 2.0533811752411, 'energy_sensitivity': 1.5650568297098282, 'duration_robustness': 0.000647990125988373, 'wait_decay': 0.47616672321832887, 'uncertainty_gate_threshold': 0.9819500743103717, 'slack_pressure_gate_steepness': 7.303317088552033, 'remaining_work_weight': 1.996821815198908, 'wait_saturation_offset': 9.868436895864314e-07, 'energy_uncertainty_interaction': 0.7531473944491167, 'safe_slack_energy_disable_threshold': 0.23965526930509262}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': 'ae8363dc8f942fe809fb80180100b4c9fde8e3db6e39b20ce17550ccc9a48e8c', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: merges Parent 2's strict DDL-dominant energy disable gate and logistic starvation relief
       with Parent 1's robust slack-margin-ratio-based urgency detection and successor-release interaction.
       Novel improvement: dual-gated energy modulation — combines both safe_slack_mask AND uncertainty-aware activation
       to avoid over-suppression under high uncertainty even with moderate slack; ensures risk-aware energy tradeoffs
       only when both feasibility and uncertainty conditions are jointly satisfied."""
    eps = 5.3385149082474506e-05
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def robust_normalize(x):
        x = np.asarray(x, dtype=float)
        center = np.median(x) if N > 1 else x[0]
        spread = np.median(np.abs(x - center)) if N > 1 else np.abs(x[0] - center) + eps
        return (x - center) / (spread + eps)
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    slack_margin_ratio = np.abs(slack) / (min_exec_time + min_comm_time + eps)
    ddl_urgent = np.where((slack <= 0.0) | (slack_margin_ratio < 0.23965526930509262), 1.0, 0.0)
    safe_slack_mask = norm_slack > 0.23965526930509262
    energy_enabled = np.where(safe_slack_mask, 0.0, 1.0)
    energy_modulation_gate = np.where((norm_uncert > 0.9819500743103717) & (ddl_urgent == 1.0), 1.0, 0.0)
    slack_pressure_norm = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-7.303317088552033 * (slack_pressure_norm - 1.0)))
    successor_release_impact = norm_work * norm_rank
    wait_benefit = 1.0 - np.exp(-0.47616672321832887 * (norm_wait + 9.868436895864314e-07))
    duration_risk_score = norm_duration * norm_uncert * ddl_urgent
    energy_uncert_penalty = norm_energy * norm_uncert * energy_modulation_gate * energy_enabled
    score = +np.clip(norm_slack, -2.0, 2.0) + np.clip(norm_slack ** 2.996226597663675, -2.0, 2.0) * ddl_urgent - np.clip(norm_rank * (1.0 + 2.0533811752411 * rank_gate * ddl_urgent), -2.0, 2.0) - np.clip(successor_release_impact, -2.0, 2.0) - np.clip(1.5650568297098282 * norm_energy * energy_modulation_gate * energy_enabled, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(0.000647990125988373 * duration_risk_score, -2.0, 2.0) + np.clip(0.7531473944491167 * energy_uncert_penalty, -2.0, 2.0) + np.clip(1.996821815198908 * norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
