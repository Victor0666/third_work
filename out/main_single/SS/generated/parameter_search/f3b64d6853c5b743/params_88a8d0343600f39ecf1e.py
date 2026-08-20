import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule with all numeric thresholds declared as parameters."""
    eps = 0.0005884915242668865

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
    slack_urgency = 0.5148554933787305 * np.tanh(np.clip(-norm_slack, 0, 4.337267446492937)) + 1.0900862711510702 * np.tanh(np.clip(norm_slack, 0, 1.8356669783629112))
    slack_pressure = np.clip(-norm_slack, 0, 4.337267446492937)
    rank_gate = 1.0 / (1.0 + np.exp(0.08277208732006508 * (slack_pressure - 2)))
    criticality_term = 1.412382731525337 * rank_gate * norm_rank
    coupling = np.tanh(0.3192193981581116 * norm_uncert * np.clip(-norm_slack, 0, 4.337267446492937))
    duration_fairness = 1.3969369578498374 * np.abs(norm_duration)
    wait_boost = 0.1833385859265978 * norm_wait
    work_term = -0.28100083895744776 * norm_work
    score = slack_urgency + coupling + 2.996753013640025 * norm_energy + criticality_term + duration_fairness - wait_boost + work_term
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score
