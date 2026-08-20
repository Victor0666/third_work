import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule with all numeric thresholds declared as parameters."""
    eps = 0.0006063386128984581

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
    slack_urgency = 6.416853748031306 * np.tanh(np.clip(-norm_slack, 0, 3.452279893050748)) + 0.729516295910126 * np.tanh(np.clip(norm_slack, 0, 1.580882371202952))
    slack_pressure = np.clip(-norm_slack, 0, 3.452279893050748)
    rank_gate = 1.0 / (1.0 + np.exp(0.2209905842801022 * (slack_pressure - 2)))
    criticality_term = 6.248268679635188e-06 * rank_gate * norm_rank
    coupling = np.tanh(0.5733969833256363 * norm_uncert * np.clip(-norm_slack, 0, 3.452279893050748))
    duration_fairness = 1.2438891287343805 * np.abs(norm_duration)
    wait_boost = 0.08286624685465876 * norm_wait
    work_term = -0.14020368156755186 * norm_work
    score = slack_urgency + coupling + 3.5788301160327904 * norm_energy + criticality_term + duration_fairness - wait_boost + work_term
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score
