import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule: 12 parameters, all used; no unused 'successor_release_weight'."""
    eps = 0.030187312084794856
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
        center = np.median(x_abs) if N > 1 else x_abs[0]
        spread = np.median(np.abs(x_abs - center)) if N > 1 else np.abs(x_abs[0] - center) + eps
        return (x_abs - center) / (spread + eps)
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    slack_distance = -slack
    ddl_protection_gate = 1.0 / (1.0 + np.exp(-1.0779420517462985 * (slack_distance + eps)))
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.7673758395113317
    norm_slack_penalty = robust_normalize(raw_slack_penalty)
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-3.975419845619072 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * (1.0 + 1.0345212146377176 * rank_gate)
    uncert_gate = np.where(norm_uncert > 0.20238966302622508, 1.0, 0.0)
    duration_risk_score = norm_duration * uncert_gate * ddl_protection_gate
    wait_benefit = 1.0 - np.exp(-0.1029988667754815 * (norm_wait + 5.135225028888536e-06))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate
    score = +norm_slack_penalty - boosted_rank - 0.9886202408122937 * norm_energy - norm_duration - wait_benefit + 0.38190838037669783 * duration_risk_score + 0.47169010288789565 * energy_uncert_penalty + 0.34223522608312523 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
