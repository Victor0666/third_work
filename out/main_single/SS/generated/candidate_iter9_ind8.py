import numpy as np
RULE_METADATA = {'structure_hash': '6ae3960973068fd08e5c6bd9b77b5210ac23f6ea9bd807a04f26dbc1d97e1b1f', 'parameter_schema_hash': '5ba6275c5a34b6ba2275df1f259e7f13fc11f659a291c28123d58d069aa3cb71', 'best_parameter_hash': '4459d7066ecf1298588cc0119cfe7784aedd8bb6fcdc71412f4349643a56e693', 'best_parameters': {'epsilon': 0.0014883142273054975, 'slack_penalty_exponent': 1.9670620325001291, 'criticality_scale': 1.796781203102937, 'energy_sensitivity': 0.8458247637828347, 'duration_robustness': 0.22694377064070625, 'wait_decay': 0.4516899531368158, 'uncertainty_gate_threshold': 0.042182981185955135, 'slack_pressure_gate_steepness': 4.209220372045323, 'remaining_work_weight': 1.4172487354663417, 'wait_saturation_offset': 1.9070569253984193e-08, 'energy_uncertainty_interaction': 0.47707534257666484, 'ddl_protection_gate_slope': 4.7948805644188734}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': 'e6b80b9df3f41082c2f17d4b68904a0d1f1877e437060b6de2a614af85b68f3c', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: replaces unstable tanh-based rank-slack coupling with bounded linear interpolation;
       introduces explicit decoupled duration weight to restore adaptability; clips all intermediate terms to [-2,2]
       to prevent score explosion while preserving ordinal ranking integrity."""
    eps = 0.0014883142273054975
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
        if N == 1:
            center = x_abs[0]
            spread = eps
        else:
            center = np.median(x_abs)
            spread = np.median(np.abs(x_abs - center))
        return (x_abs - center) / (spread + eps)
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.9670620325001291
    norm_slack_penalty = robust_normalize(raw_slack_penalty)
    coupled_slack = np.clip(norm_slack, -1.0, 1.0)
    rank_slack_coupling = 1.0 + 1.796781203102937 * coupled_slack
    boosted_rank = norm_rank * rank_slack_coupling
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-4.209220372045323 * (slack_pressure - 1.0)))
    boosted_rank = boosted_rank * (1.0 + 1.796781203102937 * rank_gate)
    ddl_gate = 1.0 / (1.0 + np.exp(-4.7948805644188734 * slack))
    uncert_gate = 1.0 / (1.0 + np.exp(-4.7948805644188734 * (norm_uncert - 0.042182981185955135)))
    duration_risk_score = norm_duration * uncert_gate * slack_pressure * ddl_gate
    wait_benefit = 1.0 - np.exp(-0.4516899531368158 * (norm_wait + 1.9070569253984193e-08))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    energy_slack_penalty = norm_energy * (1.0 + 1.9670620325001291 * slack_pressure) * ddl_gate
    duration_preference = 0.22694377064070625 * norm_duration * ddl_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 0.8458247637828347 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - duration_preference - np.clip(wait_benefit, -2.0, 2.0) + 0.22694377064070625 * np.clip(duration_risk_score, -2.0, 2.0) + 0.47707534257666484 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 1.9670620325001291 * np.clip(energy_slack_penalty, -2.0, 2.0) + 1.4172487354663417 * np.clip(norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
