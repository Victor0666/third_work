import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule with all numeric thresholds declared as parameters."""
    eps = 0.00035618117988626296

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
    slack_urgency = 3.3525270305795614 * np.tanh(np.clip(-norm_slack, 0, 2.452715951650087)) + 0.15142655084909867 * np.tanh(np.clip(norm_slack, 0, 0.863003789949724))
    slack_pressure = np.clip(-norm_slack, 0, 2.452715951650087)
    rank_gate = 1.0 / (1.0 + np.exp(0.0007152397875898861 * (slack_pressure - 2)))
    criticality_term = 1.1415841101956021 * rank_gate * norm_rank
    coupling = np.tanh(0.47715644291300435 * norm_uncert * np.clip(-norm_slack, 0, 2.452715951650087))
    duration_fairness = 0.6438163304943582 * np.abs(norm_duration)
    wait_boost = 0.018918171980900938 * norm_wait
    work_term = -0.10044307690768811 * norm_work
    score = slack_urgency + coupling + 2.652368132106206 * norm_energy + criticality_term + duration_fairness - wait_boost + work_term
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score
