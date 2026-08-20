import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule with all numeric thresholds declared as parameters."""
    eps = 0.0004153051598217345

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
    slack_urgency = 5.669318501225805 * np.tanh(np.clip(-norm_slack, 0, 1.0780677000883039)) + 3.3543698780059366 * np.tanh(np.clip(norm_slack, 0, 2.4712066141916633))
    slack_pressure = np.clip(-norm_slack, 0, 1.0780677000883039)
    rank_gate = 1.0 / (1.0 + np.exp(0.0005497497714675927 * (slack_pressure - 2)))
    criticality_term = 0.4882922260658855 * rank_gate * norm_rank
    coupling = np.tanh(0.021144648621563726 * norm_uncert * np.clip(-norm_slack, 0, 1.0780677000883039))
    duration_fairness = 1.2501125014028749 * np.abs(norm_duration)
    wait_boost = 0.606026799440316 * norm_wait
    work_term = -0.021801428023683164 * norm_work
    score = slack_urgency + coupling + 2.872638779575502 * norm_energy + criticality_term + duration_fairness - wait_boost + work_term
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score
