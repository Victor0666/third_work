import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule: all declared parameters used; no unused 'rank_slack_coupling'."""
    eps = 0.05140815161694347
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
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 3.1511936064438384
    norm_slack_penalty = robust_normalize(raw_slack_penalty)
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-3.577907467739498 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * (1.0 + 1.9337135287974507 * rank_gate)
    uncert_gate = np.where(norm_uncert > 0.49506506811849926, 1.0, 0.0)
    duration_risk_score = norm_duration * uncert_gate * slack_pressure
    wait_benefit = 1.0 - np.exp(-0.0734904515455709 * (norm_wait + 1.0686526977766014e-09))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate
    score = +norm_slack_penalty - boosted_rank - 0.9057292783524735 * norm_energy - norm_duration - wait_benefit + 1.6768619913466742 * duration_risk_score + 0.690124366523493 * energy_uncert_penalty + 0.07956492870537037 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
