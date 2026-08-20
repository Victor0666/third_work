import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule: exactly 12 parameters, all used, no hidden literals."""
    eps = 0.0925631441678886
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
        return x / (spread + eps)
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    slack_penalty = np.power(np.maximum(-slack, 0) + eps, 2.0886490100398456)
    slack_penalty = robust_normalize(slack_penalty)
    slack_pressure_level = np.clip(-norm_slack, 0, 2)
    rank_amplifier = 1.0 + 2.0063760051962167 * np.tanh(slack_pressure_level)
    uncert_gate = np.where(norm_uncert > 0.2911740033590771, norm_uncert, np.zeros_like(norm_uncert))
    duration_risk_adjusted = norm_duration * (1.0 + 1.1372544680528942 * uncert_gate)
    wait_benefit = 1.0 - np.exp(-0.24808141318651167 * norm_wait)
    coupled_criticality = np.clip(norm_rank * slack_pressure_level, 0, 2) * 0.42238820784997444
    score = 1.812828117613226 * slack_penalty + 0.8134455076919548 * coupled_criticality + 1.0269176166006908 * norm_energy + 0.4681527048300431 * duration_risk_adjusted - 0.3666517877301604 * wait_benefit
    score = np.nan_to_num(score, nan=np.finfo(float).max, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values in priority score'
    return score
