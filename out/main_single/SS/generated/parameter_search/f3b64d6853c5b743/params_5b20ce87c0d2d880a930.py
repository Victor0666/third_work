import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule with all numeric thresholds declared as parameters."""
    eps = 0.0009997805562887258

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
    slack_urgency = 3.168422420823271 * np.tanh(np.clip(-norm_slack, 0, 2.4592850190290534)) + 3.7037812928038942 * np.tanh(np.clip(norm_slack, 0, 2.493264223629545))
    slack_pressure = np.clip(-norm_slack, 0, 2.4592850190290534)
    rank_gate = 1.0 / (1.0 + np.exp(0.3328395274881518 * (slack_pressure - 2)))
    criticality_term = 0.48405323061175487 * rank_gate * norm_rank
    coupling = np.tanh(0.008703558961487622 * norm_uncert * np.clip(-norm_slack, 0, 2.4592850190290534))
    duration_fairness = 0.7525069937161724 * np.abs(norm_duration)
    wait_boost = 0.175568599848068 * norm_wait
    work_term = -4.248493786663726e-05 * norm_work
    score = slack_urgency + coupling + 3.359949335879465 * norm_energy + criticality_term + duration_fairness - wait_boost + work_term
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score
