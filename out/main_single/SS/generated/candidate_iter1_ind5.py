import numpy as np
RULE_METADATA = {'structure_hash': 'b2dc6a8ba9a5b6771715fd289d0985d2cc0814f234ee3a63d56b3b800c4f0a12', 'parameter_schema_hash': 'f8ed840ec458502239a577d257dcf3a14d0a631847cc188bf153a5e577541229', 'best_parameter_hash': '8c51106f809150c0277db4fc294fba1c64fde33f2be5d3b30d48083a9cf30e2f', 'best_parameters': {'epsilon': 2.9265104238251944e-06, 'slack_penalty_exponent': 1.660102963062007, 'criticality_scale': 1.39953073152212, 'energy_sensitivity': 1.9843204687439884, 'duration_robustness': 1.5803271157019498, 'wait_decay': 0.528022712800696, 'uncertainty_gate_threshold': 0.3301385660239994, 'rank_slack_coupling': 0.7012175838519186, 'remaining_work_weight': 0.20488351549992972, 'wait_saturation_offset': 1.4904158807965674e-05}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': 'e51083d249995c12daa391edba62dbc9314afd84b7aaab6a2bd80e29bacadeb5', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule: all numeric coefficients declared in PARAMETER_SCHEMA."""
    eps = 2.9265104238251944e-06
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
        x = np.abs(x)
        center = np.median(x)
        scale = np.median(np.abs(x - center)) + eps
        return (x - center) / scale
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.660102963062007
    norm_slack_penalty = robust_normalize(raw_slack_penalty)
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-0.7012175838519186 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * (1.0 + 1.39953073152212 * rank_gate)
    uncert_gate = np.where(norm_uncert > 0.3301385660239994, 1.0, 0.0)
    duration_risk_score = norm_duration * uncert_gate * slack_pressure
    wait_benefit = 1.0 - np.exp(-0.528022712800696 * (norm_wait + 1.4904158807965674e-05))
    score = +norm_slack_penalty - boosted_rank - 1.9843204687439884 * norm_energy - norm_duration - wait_benefit + 1.5803271157019498 * duration_risk_score + 0.20488351549992972 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
