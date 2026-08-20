import numpy as np
RULE_METADATA = {'structure_hash': 'f3b64d6853c5b743e7f6fb047dabc6cc8cb79638d2d8d8450d17ba92c2c5e7a8', 'parameter_schema_hash': 'ee2fad00b985e26bb68e393c45c8da729f166816cdd1f05d380073cf8026cd2a', 'best_parameter_hash': '8ec7cfea556e4d29621e344128bf604cf10c8922f1401f0ae0832aa6984f5cca', 'best_parameters': {'epsilon': 0.0005884915242668865, 'slack_risk_penalty': 0.5148554933787305, 'slack_urgency_gain': 1.0900862711510702, 'energy_scale': 2.996753013640025, 'criticality_weight': 1.412382731525337, 'wait_bias': 0.1833385859265978, 'uncertainty_slack_coupling': 0.3192193981581116, 'duration_fairness': 1.3969369578498374, 'rank_slack_gate': 0.08277208732006508, 'slack_clip_upper': 4.337267446492937, 'slack_clip_upper_pos': 1.8356669783629112, 'work_weight': 0.24100083895744778}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '9390ba55708b6485af468f3cc10451ccf9c0456cbd24661e7cbfee10a70e7dc8', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule with all numeric thresholds declared as parameters."""
    eps = 0.0005884915242668865

    def robust_normalize(x):
        x = np.asarray(x, dtype=float)
        center = np.median(x)
        scale = np.median(np.abs(x - center)) + eps
        return (x - center) / scale
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    slack_urgency = 0.5148554933787305 * np.tanh(np.clip(-norm_slack, 0, 4.337267446492937)) + 1.0900862711510702 * np.tanh(np.clip(norm_slack, 0, 1.8356669783629112))
    slack_pressure = np.clip(-norm_slack, 0, 4.337267446492937)
    rank_gate = 1.0 / (1.0 + np.exp(0.08277208732006508 * (slack_pressure - 2)))
    criticality_term = 1.412382731525337 * rank_gate * norm_rank
    coupling = np.tanh(0.3192193981581116 * norm_uncert * np.clip(-norm_slack, 0, 4.337267446492937))
    duration_fairness = 1.3969369578498374 * np.abs(norm_duration)
    wait_boost = 0.1833385859265978 * norm_wait
    work_term = -0.24100083895744778 * norm_work
    score = slack_urgency + coupling + 2.996753013640025 * norm_energy + criticality_term + duration_fairness - wait_boost + work_term
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score
