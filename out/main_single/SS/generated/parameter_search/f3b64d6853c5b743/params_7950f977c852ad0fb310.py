import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule with all numeric thresholds declared as parameters."""
    eps = 0.0002696809055431836

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
    slack_urgency = 9.765124505137338 * np.tanh(np.clip(-norm_slack, 0, 5.057618154577417)) + 1.2498125056625802 * np.tanh(np.clip(norm_slack, 0, 1.018984827639492))
    slack_pressure = np.clip(-norm_slack, 0, 5.057618154577417)
    rank_gate = 1.0 / (1.0 + np.exp(0.17511188164969388 * (slack_pressure - 2)))
    criticality_term = 2.8420998856426882 * rank_gate * norm_rank
    coupling = np.tanh(1.1570137836808438 * norm_uncert * np.clip(-norm_slack, 0, 5.057618154577417))
    duration_fairness = 1.028207926510854 * np.abs(norm_duration)
    wait_boost = 0.42860687884066073 * norm_wait
    work_term = -0.02937580542866395 * norm_work
    score = slack_urgency + coupling + 0.08292764790274318 * norm_energy + criticality_term + duration_fairness - wait_boost + work_term
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score
