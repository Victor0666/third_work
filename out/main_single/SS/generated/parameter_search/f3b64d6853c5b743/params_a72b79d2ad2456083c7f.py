import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule with all numeric thresholds declared as parameters."""
    eps = 3.118684151936646e-06

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
    slack_urgency = 6.116332796701028 * np.tanh(np.clip(-norm_slack, 0, 8.41943001762499)) + 0.161245059567459 * np.tanh(np.clip(norm_slack, 0, 2.6806689701883153))
    slack_pressure = np.clip(-norm_slack, 0, 8.41943001762499)
    rank_gate = 1.0 / (1.0 + np.exp(0.6037570251750959 * (slack_pressure - 2)))
    criticality_term = 0.748043362543644 * rank_gate * norm_rank
    coupling = np.tanh(0.7943302487344235 * norm_uncert * np.clip(-norm_slack, 0, 8.41943001762499))
    duration_fairness = 0.06411158806889264 * np.abs(norm_duration)
    wait_boost = 0.16263657655706654 * norm_wait
    work_term = -0.062194736549010095 * norm_work
    score = slack_urgency + coupling + 0.1300770821102356 * norm_energy + criticality_term + duration_fairness - wait_boost + work_term
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score
