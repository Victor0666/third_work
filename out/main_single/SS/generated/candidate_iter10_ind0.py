import numpy as np
RULE_METADATA = {'structure_hash': 'd202a0af3c2eadee2443263f0455802ebfd695f34ddbf504a48c705f2aadacfd', 'parameter_schema_hash': '56914beb76d428c4c9b5658c7028d9b6d68980e2336f457f49bed530c1f41fb5', 'best_parameter_hash': 'd9ea9c101c054c04010a45990bc467745ad4181dbbe8447abc17844c1dba9432', 'best_parameters': {'epsilon': 0.02031086128955383, 'slack_penalty_exponent': 3.6207569935337633, 'criticality_scale': 1.4672169004499667, 'energy_sensitivity': 0.42270026220184453, 'duration_robustness': 0.03764108117780722, 'wait_decay': 0.8460891212802435, 'uncertainty_gate_threshold': 0.5545469152061369, 'slack_pressure_gate_steepness': 5.873936830895824, 'remaining_work_weight': 0.22222427299588368, 'energy_uncertainty_interaction': 0.5459754820566909, 'ddl_urgency_boost': 8.317436846098955}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '998c3491117a69ef7e8da0129f8b2f7e5d22a3dbd8d2d385570b5654e500c1c7', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule: adds conditional DDL protection gate, successor-release interaction, 
    uses clipped tanh for bounded anti-starvation, and declares all numeric coefficients as parameters."""
    eps = 0.02031086128955383
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
        center = np.median(x) if N > 1 else x[0]
        spread = np.median(np.abs(x - center)) + eps if N > 1 else eps
        return (x - center) / (spread + eps)
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    ddl_urgent_mask = (slack <= eps).astype(float)
    raw_slack_penalty = np.where(slack < 0, (-slack) ** 3.6207569935337633, 0.0)
    norm_slack_penalty = robust_normalize(raw_slack_penalty)
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-5.873936830895824 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * (1.0 + 1.4672169004499667 * rank_gate)
    successor_release_score = norm_rank * norm_work
    uncert_gate = np.where(norm_uncert > 0.5545469152061369, 1.0, 0.0)
    duration_risk_score = norm_duration * uncert_gate * slack_pressure
    wait_benefit = np.tanh(0.8460891212802435 * norm_wait)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate
    score = +norm_slack_penalty - ddl_urgent_mask * 8.317436846098955 - boosted_rank - 0.42270026220184453 * norm_energy - norm_duration - wait_benefit + 0.03764108117780722 * duration_risk_score + 0.5459754820566909 * energy_uncert_penalty + 0.22222427299588368 * successor_release_score
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
