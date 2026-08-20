import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule with all numeric thresholds declared as parameters."""
    eps = 0.0003553412886012764

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
    slack_urgency = 7.78019984520332 * np.tanh(np.clip(-norm_slack, 0, 2.9965178318536125)) + 0.976355029591958 * np.tanh(np.clip(norm_slack, 0, 2.4977769005914405))
    slack_pressure = np.clip(-norm_slack, 0, 2.9965178318536125)
    rank_gate = 1.0 / (1.0 + np.exp(0.178726910586888 * (slack_pressure - 2)))
    criticality_term = 0.22492200815084828 * rank_gate * norm_rank
    coupling = np.tanh(0.12563745802284212 * norm_uncert * np.clip(-norm_slack, 0, 2.9965178318536125))
    duration_fairness = 1.354279688521529e-05 * np.abs(norm_duration)
    wait_boost = 0.2596211545524164 * norm_wait
    work_term = -0.011939165111111788 * norm_work
    score = slack_urgency + coupling + 1.6905720707725618 * norm_energy + criticality_term + duration_fairness - wait_boost + work_term
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score
