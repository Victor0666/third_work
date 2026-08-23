import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule: replaces hardcoded 75/25 with tunable percentiles.
    Uses robust IQR-normalized features, piecewise slack penalty, and risk-modulated criticality.
    Smaller score = higher priority."""
    eps = 4.803365134971875e-07
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
        q75_val = 71.53319671940598
        q25_val = 29.016064955636256
        q75, q25 = np.percentile(x, [q75_val, q25_val], method='midpoint')
        iqr = q75 - q25
        mad = np.mean(np.abs(x - np.median(x)))
        scale = np.where(iqr > eps, iqr, mad + eps)
        center = np.median(x)
        return (x - center) / (scale + eps)
    slack_abs = np.abs(slack)
    slack_sign = np.sign(slack)
    slack_powered = np.where(slack >= 0, slack_abs ** 0.7017951369385345, -slack_abs ** 0.7017951369385345)
    slack_pressure = np.where(slack < 0, slack_powered * 7.0494966885244805, slack_powered * 0.4483926408349846)
    energy_norm = robust_normalize(min_incremental_energy)
    energy_score = 0.6002659315423204 * energy_norm
    duration = min_exec_time + min_comm_time
    duration_norm = robust_normalize(duration)
    duration_score = 0.5129308245123553 * duration_norm
    risk_signal = np.maximum(-slack, 0.0) * uncertainty
    risk_clamped = np.tanh(risk_signal / np.maximum(np.mean(np.abs(risk_signal)) + eps, eps))
    rank_score = -upward_rank * (1.0 + 2.0159028282885227 * risk_clamped)
    wait_boost = 1.0 - np.exp(-0.45155089187335196 * ready_wait_time)
    wait_score = -wait_boost
    unc_norm = robust_normalize(uncertainty)
    unc_slack_interaction = 0.11589488630346395 * unc_norm * slack_pressure
    score = slack_pressure + energy_score + duration_score + rank_score + wait_score + unc_slack_interaction
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score
