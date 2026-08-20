import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule with all numeric thresholds declared as parameters."""
    eps = 0.0003725884605700906

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
    slack_urgency = 1.1811198145904567 * np.tanh(np.clip(-norm_slack, 0, 5.753961896817412)) + 1.2780034883023423 * np.tanh(np.clip(norm_slack, 0, 1.3444019491402397))
    slack_pressure = np.clip(-norm_slack, 0, 5.753961896817412)
    rank_gate = 1.0 / (1.0 + np.exp(0.32382818960510346 * (slack_pressure - 2)))
    criticality_term = 0.04035366006035996 * rank_gate * norm_rank
    coupling = np.tanh(1.2412148536954493 * norm_uncert * np.clip(-norm_slack, 0, 5.753961896817412))
    duration_fairness = 1.0716979322983873 * np.abs(norm_duration)
    wait_boost = 0.24145918090483864 * norm_wait
    work_term = -0.4169928412140421 * norm_work
    score = slack_urgency + coupling + 3.7895429077469487 * norm_energy + criticality_term + duration_fairness - wait_boost + work_term
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score
