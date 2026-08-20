import numpy as np
RULE_METADATA = {'structure_hash': '77e79a663605f2d2eef922bb4a1c4c63c6bae2e9b8262236c812ab9233b0feba', 'parameter_schema_hash': 'eac839bd2f03ff7250f20ddbcac274b2f564939e71c479c95c2cfe3723a245fd', 'best_parameter_hash': '0f9286dd40955340f730f9aa17b8b18abbc357071f90e0e01650fbcb8b60a344', 'best_parameters': {'epsilon': 0.0018233143382804563, 'slack_penalty_exponent': 3.1456982549622126, 'criticality_scale': 1.177769039043172, 'energy_sensitivity': 0.6499638459261315, 'duration_robustness': 1.379605765645357, 'wait_decay': 0.9382724840065622, 'uncertainty_gate_threshold': 0.24717072245475163, 'slack_gate_width': 1.3751677754892233, 'energy_slack_interaction': 0.6015392175851265, 'rank_slack_coupling': 0.3272534552715318, 'uncertainty_smoothness': 1.1329481671609105}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '1d338b3f314c1596ec50408a4b88a0bf69d75b15f431b859795dd9cdbeca98ef', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: restores rank_slack_coupling for hard deadline feasibility; replaces binary uncertainty gate with smooth sigmoidal activation; retains improved energy-slack interaction and saturating starvation relief."""
    eps = 0.0018233143382804563
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
        x_abs = np.abs(x)
        center = np.median(x_abs)
        scale = np.median(np.abs(x_abs - center)) + eps
        return (x_abs - center) / scale
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    slack_pressure = np.power(np.clip(-norm_slack, 0.0, None), 3.1456982549622126)
    rank_gate = np.clip(1.0 - norm_slack / (1.3751677754892233 + eps), 0.0, 1.0)
    boosted_rank = norm_rank * (1.0 + 1.177769039043172 * rank_gate)
    uncertainty_activation = 1.0 / (1.0 + np.exp(-1.1329481671609105 * (norm_uncert - 0.24717072245475163)))
    duration_risk_interaction = norm_duration * uncertainty_activation * 1.379605765645357
    wait_benefit = 1.0 - np.exp(-0.9382724840065622 * (norm_wait + eps))
    energy_slack_penalty = norm_energy * (1.0 + 0.6015392175851265 * slack_pressure)
    coupled_rank_reward = norm_rank * (1.0 + 0.3272534552715318 * slack_pressure)
    score = +slack_pressure + energy_slack_penalty + 0.6499638459261315 * norm_energy - coupled_rank_reward - wait_benefit + duration_risk_interaction
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
