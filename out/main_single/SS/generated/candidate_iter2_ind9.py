import numpy as np
RULE_METADATA = {'structure_hash': 'ddbf66beaaca56f7f1d357b1571b9e559d233e782b346816fb9fb88be1ae592a', 'parameter_schema_hash': '8edc0070900d41259701e2d5179abbeeb765a68aa2ee39e3a463ed4fe4959ed4', 'best_parameter_hash': '3e8c23f170406daa8b7d373521742ad39de594a241b9b920bb90f26866286e01', 'best_parameters': {'epsilon': 0.028120940384698232, 'slack_penalty_exponent': 2.040862907362441, 'criticality_scale': 1.2678893064812344, 'energy_sensitivity': 1.2777166887435327, 'duration_robustness': 0.2510786195766228, 'wait_decay': 0.017295236242829068, 'uncertainty_gate_threshold': 0.5456396809702606, 'slack_pressure_gate_steepness': 3.827711036827819, 'remaining_work_weight': 0.8668804611038748, 'wait_saturation_offset': 3.296113552016036e-08, 'energy_uncertainty_interaction': 0.5806548650345438}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '7b2db0bfc86b9659c70f3d3465a331cd3cf32991d515fa692026476f7367332b', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule: all declared parameters used; no unused 'rank_slack_coupling'."""
    eps = 0.028120940384698232
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
        spread = np.median(np.abs(x - center)) if N > 1 else np.abs(x[0] - center) + eps
        return (x - center) / (spread + eps)
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.040862907362441
    norm_slack_penalty = robust_normalize(raw_slack_penalty)
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-3.827711036827819 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * (1.0 + 1.2678893064812344 * rank_gate)
    uncert_gate = np.where(norm_uncert > 0.5456396809702606, 1.0, 0.0)
    duration_risk_score = norm_duration * uncert_gate * slack_pressure
    wait_benefit = 1.0 - np.exp(-0.017295236242829068 * (norm_wait + 3.296113552016036e-08))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate
    score = +norm_slack_penalty - boosted_rank - 1.2777166887435327 * norm_energy - norm_duration - wait_benefit + 0.2510786195766228 * duration_risk_score + 0.5806548650345438 * energy_uncert_penalty + 0.8668804611038748 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
