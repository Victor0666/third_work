import numpy as np
RULE_METADATA = {'structure_hash': '929bd164ae573002b3a3be289f9ddbc683c66ed813f2deade8526ecfdf13576d', 'parameter_schema_hash': '66c6739241b4deb01f7a07475ab9537dd3bb92470145c7d19b553362d66899b8', 'best_parameter_hash': '1d4a03398ee622ac50845b2397eb494af93e74e975c9b1a6bff6257308b2b55d', 'best_parameters': {'epsilon': 4.433081659558651e-08, 'slack_penalty_scale': 2.746516285563475, 'slack_sensitivity': 1.5271939447892613, 'energy_norm_factor': 0.10344479698512113, 'criticality_boost': 1.7572025975211891, 'wait_decay': 0.3226001344937595, 'uncertainty_slack_interaction': 0.3124069000422247, 'duration_balance': 0.3747405297885284, 'slack_positive_weight': 0.16280571627251658, 'q75_percentile': 72.80235158387859, 'q25_percentile': 29.67447387740997}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': 'b624808e858c6d6624161c984629449a396a73fe1e91b0f7aa0ff8783bb37d9e', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule: replaces hardcoded 75/25 with tunable percentiles.
    Uses robust IQR-normalized features, piecewise slack penalty, and risk-modulated criticality.
    Smaller score = higher priority."""
    eps = 4.433081659558651e-08
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)

    def robust_normalize(x):
        x = np.asarray(x)
        q75_val = 72.80235158387859
        q25_val = 29.67447387740997
        q75, q25 = np.percentile(x, [q75_val, q25_val], method='midpoint')
        iqr = q75 - q25
        mad = np.mean(np.abs(x - np.median(x)))
        scale = np.where(iqr > eps, iqr, mad + eps)
        center = np.median(x)
        return (x - center) / (scale + eps)
    slack_abs = np.abs(slack)
    slack_sign = np.sign(slack)
    slack_powered = np.where(slack >= 0, slack_abs ** 1.5271939447892613, -slack_abs ** 1.5271939447892613)
    slack_pressure = np.where(slack < 0, slack_powered * 2.746516285563475, slack_powered * 0.16280571627251658)
    energy_norm = robust_normalize(min_incremental_energy)
    energy_score = 0.10344479698512113 * energy_norm
    duration = min_exec_time + min_comm_time
    duration_norm = robust_normalize(duration)
    duration_score = 0.3747405297885284 * duration_norm
    risk_signal = np.maximum(-slack, 0.0) * uncertainty
    risk_clamped = np.tanh(risk_signal / np.maximum(np.mean(np.abs(risk_signal)) + eps, eps))
    rank_score = -upward_rank * (1.0 + 1.7572025975211891 * risk_clamped)
    wait_boost = 1.0 - np.exp(-0.3226001344937595 * ready_wait_time)
    wait_score = -wait_boost
    unc_norm = robust_normalize(uncertainty)
    unc_slack_interaction = 0.3124069000422247 * unc_norm * slack_pressure
    score = slack_pressure + energy_score + duration_score + rank_score + wait_score + unc_slack_interaction
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score
