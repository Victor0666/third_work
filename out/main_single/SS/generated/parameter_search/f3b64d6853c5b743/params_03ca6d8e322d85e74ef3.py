import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule with all numeric thresholds declared as parameters."""
    eps = 4.837105884771383e-07

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
    slack_urgency = 3.708407187103054 * np.tanh(np.clip(-norm_slack, 0, 3.5775108952883397)) + 2.7309773178901384 * np.tanh(np.clip(norm_slack, 0, 2.27258566826529))
    slack_pressure = np.clip(-norm_slack, 0, 3.5775108952883397)
    rank_gate = 1.0 / (1.0 + np.exp(0.2164169165050311 * (slack_pressure - 2)))
    criticality_term = 1.434757075237916 * rank_gate * norm_rank
    coupling = np.tanh(1.3888536691533258 * norm_uncert * np.clip(-norm_slack, 0, 3.5775108952883397))
    duration_fairness = 0.04685282081789195 * np.abs(norm_duration)
    wait_boost = 0.0020154384669058825 * norm_wait
    work_term = -0.00037650257059870974 * norm_work
    score = slack_urgency + coupling + 2.5788037675603874 * norm_energy + criticality_term + duration_fairness - wait_boost + work_term
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score
