import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule with all numeric thresholds declared as parameters."""
    eps = 2.019847562284639e-05

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
    slack_urgency = 3.6911467937529476 * np.tanh(np.clip(-norm_slack, 0, 3.7169935040214654)) + 2.212176254929559 * np.tanh(np.clip(norm_slack, 0, 1.99848433761223))
    slack_pressure = np.clip(-norm_slack, 0, 3.7169935040214654)
    rank_gate = 1.0 / (1.0 + np.exp(0.01964657944018654 * (slack_pressure - 2)))
    criticality_term = 0.13102087833776263 * rank_gate * norm_rank
    coupling = np.tanh(0.8237174124914852 * norm_uncert * np.clip(-norm_slack, 0, 3.7169935040214654))
    duration_fairness = 0.6928288958252428 * np.abs(norm_duration)
    wait_boost = 0.48164883340828885 * norm_wait
    work_term = -0.14986517500058955 * norm_work
    score = slack_urgency + coupling + 0.6835190764725375 * norm_energy + criticality_term + duration_fairness - wait_boost + work_term
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score
