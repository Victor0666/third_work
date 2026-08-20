import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule with all numeric thresholds declared as parameters."""
    eps = 0.0009153947340266438

    def robust_normalize(x):
        x = np.asarray(x, dtype=float)
        center = np.median(x)
        scale = np.median(np.abs(x - center)) + eps
        return (x - center) / scale
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    slack_urgency = 6.968026498457884 * np.tanh(np.clip(-norm_slack, 0, 3.003445958124377)) + 0.2970670050642985 * np.tanh(np.clip(norm_slack, 0, 2.768145453457668))
    slack_pressure = np.clip(-norm_slack, 0, 3.003445958124377)
    rank_gate = 1.0 / (1.0 + np.exp(0.12336106481357288 * (slack_pressure - 2)))
    criticality_term = 0.7345936738063781 * rank_gate * norm_rank
    coupling = np.tanh(0.3309394904661865 * norm_uncert * np.clip(-norm_slack, 0, 3.003445958124377))
    duration_fairness = 0.5694091170337485 * np.abs(norm_duration)
    wait_boost = 0.014219184740965528 * norm_wait
    work_term = -0.16589336267221336 * norm_work
    score = slack_urgency + coupling + 2.3091721169057497 * norm_energy + criticality_term + duration_fairness - wait_boost + work_term
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score
