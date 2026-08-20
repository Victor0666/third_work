import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule: all declared parameters used; no unused 'rank_slack_coupling'."""
    eps = 0.005934284682361182
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
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.4914438271245647
    norm_slack_penalty = robust_normalize(raw_slack_penalty)
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-1.231669138769263 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * (1.0 + 1.18460137001378 * rank_gate)
    uncert_gate = np.where(norm_uncert > 0.03178764579049887, 1.0, 0.0)
    duration_risk_score = norm_duration * uncert_gate * slack_pressure
    wait_benefit = 1.0 - np.exp(-0.309709140645166 * (norm_wait + 2.563033497744147e-05))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate
    score = +norm_slack_penalty - boosted_rank - 1.1372252008810877 * norm_energy - norm_duration - wait_benefit + 0.1954520755558039 * duration_risk_score + 0.33666788982358264 * energy_uncert_penalty + 0.48136470029793293 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
