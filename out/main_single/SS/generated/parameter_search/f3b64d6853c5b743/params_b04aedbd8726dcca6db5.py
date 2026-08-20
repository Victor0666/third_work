import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule with all numeric thresholds declared as parameters."""
    eps = 0.00014837810834173143

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
    slack_urgency = 5.002858902203931 * np.tanh(np.clip(-norm_slack, 0, 2.749338016942046)) + 1.5958995094314776 * np.tanh(np.clip(norm_slack, 0, 2.3144940857720036))
    slack_pressure = np.clip(-norm_slack, 0, 2.749338016942046)
    rank_gate = 1.0 / (1.0 + np.exp(0.29905418589362415 * (slack_pressure - 2)))
    criticality_term = 0.2915091637514865 * rank_gate * norm_rank
    coupling = np.tanh(0.4635294888460741 * norm_uncert * np.clip(-norm_slack, 0, 2.749338016942046))
    duration_fairness = 1.187270308672074 * np.abs(norm_duration)
    wait_boost = 0.18800329843563968 * norm_wait
    work_term = -0.4490462416398041 * norm_work
    score = slack_urgency + coupling + 4.545629105083904 * norm_energy + criticality_term + duration_fairness - wait_boost + work_term
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score
