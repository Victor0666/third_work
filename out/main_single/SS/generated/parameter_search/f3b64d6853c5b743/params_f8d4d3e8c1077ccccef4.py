import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule with all numeric thresholds declared as parameters."""
    eps = 6.8188398546642074e-06

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
    slack_urgency = 3.9362862044184777 * np.tanh(np.clip(-norm_slack, 0, 8.904029700213957)) + 1.0191137841680709 * np.tanh(np.clip(norm_slack, 0, 1.6085324131888983))
    slack_pressure = np.clip(-norm_slack, 0, 8.904029700213957)
    rank_gate = 1.0 / (1.0 + np.exp(0.025940862245968026 * (slack_pressure - 2)))
    criticality_term = 3.079507124622549 * rank_gate * norm_rank
    coupling = np.tanh(0.7805130873187716 * norm_uncert * np.clip(-norm_slack, 0, 8.904029700213957))
    duration_fairness = 1.1381229702302005 * np.abs(norm_duration)
    wait_boost = 0.11524195977147622 * norm_wait
    work_term = -0.26591873469223803 * norm_work
    score = slack_urgency + coupling + 1.157562675585746 * norm_energy + criticality_term + duration_fairness - wait_boost + work_term
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score
