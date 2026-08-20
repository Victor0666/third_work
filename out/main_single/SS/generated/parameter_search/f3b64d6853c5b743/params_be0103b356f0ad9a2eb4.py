import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule with all numeric thresholds declared as parameters."""
    eps = 0.0001898121927004307

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
    slack_urgency = 6.256638173980453 * np.tanh(np.clip(-norm_slack, 0, 2.785433780406514)) + 1.6952291321509592 * np.tanh(np.clip(norm_slack, 0, 2.283239665389202))
    slack_pressure = np.clip(-norm_slack, 0, 2.785433780406514)
    rank_gate = 1.0 / (1.0 + np.exp(0.010343487780115027 * (slack_pressure - 2)))
    criticality_term = 1.0039926688913132 * rank_gate * norm_rank
    coupling = np.tanh(0.02024551469001467 * norm_uncert * np.clip(-norm_slack, 0, 2.785433780406514))
    duration_fairness = 0.7544085031228152 * np.abs(norm_duration)
    wait_boost = 0.35976737583072915 * norm_wait
    work_term = -0.0764429434062807 * norm_work
    score = slack_urgency + coupling + 3.1484603737870676 * norm_energy + criticality_term + duration_fairness - wait_boost + work_term
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score
