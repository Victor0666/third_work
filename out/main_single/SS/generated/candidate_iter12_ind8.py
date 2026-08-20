import numpy as np
RULE_METADATA = {'structure_hash': 'f629d27bc243379293e2bcdafb502686b47d931a01b31d16ab991c8b3b3ee80a', 'parameter_schema_hash': 'd05b1086cf35f301af0b7490a3e74c153d952d286f9b635b18c89fa9eda0cfc1', 'best_parameter_hash': '2d87be9783f86bcb2b908b3917abc74035ed34e0d07c9b396ae8c8388cee60e7', 'best_parameters': {'epsilon': 0.0066540928038742635, 'slack_penalty_exponent': 1.256597067252815, 'criticality_scale': 0.9367297604197168, 'energy_sensitivity': 0.8020341662651104, 'duration_robustness': 0.2568895858348938, 'wait_decay': 0.8892868418235393, 'uncertainty_gate_threshold': 0.2850583405336597, 'slack_pressure_gate_steepness': 2.74867482967456, 'remaining_work_weight': 0.8180148666469828, 'wait_saturation_offset': 4.192004088136645e-06, 'energy_uncertainty_interaction': 0.5832573322657771, 'safe_slack_energy_disable_threshold': 0.6413978636498817}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': 'ef70d94dbb0d160d730d3c6703a94bc6cec170410e0c2eb20aa604bfef7ef9dd', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: restores uncertainty-aware energy gating, replaces soft modulation with hard DDL-dominant energy disabling,
       and strengthens deadline feasibility via strict slack-based conditional logic.
       Key structural change: introduces 'safe_slack_energy_disable_threshold' to fully zero out energy terms when slack is sufficiently positive,
       enforcing strict priority ordering where deadline feasibility always dominates energy minimization."""
    eps = 0.0066540928038742635
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
    ddl_urgent = np.where((slack <= 0.0) | (slack_margin_ratio < 0.6413978636498817), 1.0, 0.0)
    safe_slack_mask = norm_slack > 0.6413978636498817
    energy_enabled = np.where(safe_slack_mask, 0.0, 1.0)
    successor_release_impact = norm_work * norm_rank
    energy_modulation_gate = np.where((norm_uncert > 0.2850583405336597) & (ddl_urgent == 1.0), 1.0, 0.0)
    slack_pressure_norm = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-2.74867482967456 * (slack_pressure_norm - 1.0)))
    boosted_rank = norm_rank * (1.0 + 0.9367297604197168 * rank_gate * ddl_urgent)
    wait_benefit = 1.0 - np.exp(-0.8892868418235393 * (norm_wait + 4.192004088136645e-06))
    duration_risk_score = norm_duration * norm_uncert * ddl_urgent
    energy_uncert_penalty = norm_energy * norm_uncert * energy_modulation_gate * energy_enabled
    score = +np.clip(norm_slack, -2.0, 2.0) + np.clip(norm_slack ** 1.256597067252815, -2.0, 2.0) * ddl_urgent - np.clip(boosted_rank, -2.0, 2.0) - np.clip(successor_release_impact, -2.0, 2.0) - np.clip(norm_energy * 0.8020341662651104 * energy_modulation_gate * energy_enabled, -2.0, 2.0) - np.clip(norm_duration * (1.0 - ddl_urgent), -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(0.2568895858348938 * duration_risk_score, -2.0, 2.0) + np.clip(0.5832573322657771 * energy_uncert_penalty, -2.0, 2.0) + np.clip(0.8180148666469828 * norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
