import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule with all numeric thresholds declared as parameters."""
    eps = 5.856965734824187e-05

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
    slack_urgency = 0.5725974294964608 * np.tanh(np.clip(-norm_slack, 0, 1.7570703991628158)) + 0.4422731954824314 * np.tanh(np.clip(norm_slack, 0, 2.355619583508277))
    slack_pressure = np.clip(-norm_slack, 0, 1.7570703991628158)
    rank_gate = 1.0 / (1.0 + np.exp(0.4548376927546186 * (slack_pressure - 2)))
    criticality_term = 0.25703512539234474 * rank_gate * norm_rank
    coupling = np.tanh(1.0351913808893627 * norm_uncert * np.clip(-norm_slack, 0, 1.7570703991628158))
    duration_fairness = 1.0434235573376294 * np.abs(norm_duration)
    wait_boost = 0.12154919855588872 * norm_wait
    work_term = -0.6464336943001225 * norm_work
    score = slack_urgency + coupling + 1.8674011785430986 * norm_energy + criticality_term + duration_fairness - wait_boost + work_term
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score
