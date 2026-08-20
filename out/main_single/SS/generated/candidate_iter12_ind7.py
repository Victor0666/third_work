import numpy as np
RULE_METADATA = {'structure_hash': '64ec74963fdae7d9b3f6ad293a01d41ba74142426d471dab1e88e181c542aa53', 'parameter_schema_hash': '5ba6275c5a34b6ba2275df1f259e7f13fc11f659a291c28123d58d069aa3cb71', 'best_parameter_hash': '6fd88aee77d9bd9ba49652d1224ca86d11a5eb5fb78e95db9d492bd596da3c4d', 'best_parameters': {'epsilon': 0.004471698489018286, 'slack_penalty_exponent': 1.6842441303164084, 'criticality_scale': 1.4152777959749354, 'energy_sensitivity': 0.7595765898592081, 'duration_robustness': 0.003541509077342696, 'wait_decay': 0.055420178507395806, 'uncertainty_gate_threshold': 0.3315600396857239, 'slack_pressure_gate_steepness': 1.840279982589781, 'remaining_work_weight': 0.15425288570580742, 'wait_saturation_offset': 1.4774789017680615e-05, 'energy_uncertainty_interaction': 0.7025646308764023, 'ddl_protection_gate_slope': 3.2975101980306745}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': 'f8cd4e2d7052f8e3bda89c0806582dd76c58fbbcbf3ddda46273e07895e99a8a', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: MAD-based normalization; sign-aware slack handling;
       simplified single-sigmoid urgency gate; no redundant comm coupling; all 12 params used."""
    eps = 0.004471698489018286
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
    slack_sign = np.sign(slack)
    norm_slack_signed = slack_sign * np.clip(np.abs(norm_slack), 0.0, 2.0)
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.6842441303164084
    norm_slack_penalty = robust_normalize(raw_slack_penalty)
    slack_pressure = np.clip(-slack, 0.0, np.inf)
    if N == 1:
        urgency_center = slack_pressure[0]
        urgency_spread = eps
    else:
        urgency_center = np.median(slack_pressure)
        urgency_spread = np.median(np.abs(slack_pressure - urgency_center)) + eps
    norm_urgency = (slack_pressure - urgency_center) / (urgency_spread + eps)
    rank_gate = 1.0 / (1.0 + np.exp(-1.840279982589781 * (norm_urgency - 1.0)))
    boosted_rank = norm_rank * (1.0 + 1.4152777959749354 * rank_gate)
    ddl_gate = 1.0 / (1.0 + np.exp(-3.2975101980306745 * slack))
    uncert_gate = 1.0 / (1.0 + np.exp(-3.2975101980306745 * (norm_uncert - 0.3315600396857239)))
    duration_risk_score = norm_duration * uncert_gate * rank_gate * ddl_gate
    wait_benefit = 1.0 - np.exp(-0.055420178507395806 * (norm_wait + 1.4774789017680615e-05))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    tight_slack_mask = (slack > 0) & (slack < 0.3315600396857239 * (np.max(np.abs(slack)) + eps))
    energy_slack_penalty = norm_energy * (1.0 + 1.6842441303164084 * np.where(tight_slack_mask, 1.0, 0.0)) * ddl_gate
    duration_preference = 0.003541509077342696 * norm_duration * ddl_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 0.7595765898592081 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - duration_preference - np.clip(wait_benefit, -2.0, 2.0) + 0.003541509077342696 * np.clip(duration_risk_score, -2.0, 2.0) + 0.7025646308764023 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 1.6842441303164084 * np.clip(energy_slack_penalty, -2.0, 2.0) + 0.15425288570580742 * np.clip(norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
