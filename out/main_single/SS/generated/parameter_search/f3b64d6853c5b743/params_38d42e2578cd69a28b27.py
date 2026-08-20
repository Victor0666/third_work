import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule with all numeric thresholds declared as parameters."""
    eps = 4.79955501166907e-06

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
    slack_urgency = 1.5338485225405056 * np.tanh(np.clip(-norm_slack, 0, 5.318676307642834)) + 1.0023626229154483 * np.tanh(np.clip(norm_slack, 0, 1.5805763609041539))
    slack_pressure = np.clip(-norm_slack, 0, 5.318676307642834)
    rank_gate = 1.0 / (1.0 + np.exp(0.3819450620182717 * (slack_pressure - 2)))
    criticality_term = 1.1925195187634583 * rank_gate * norm_rank
    coupling = np.tanh(0.9765723305592567 * norm_uncert * np.clip(-norm_slack, 0, 5.318676307642834))
    duration_fairness = 0.042942385476732146 * np.abs(norm_duration)
    wait_boost = 0.3484631236169745 * norm_wait
    work_term = -0.4715450406290849 * norm_work
    score = slack_urgency + coupling + 0.2273913916909331 * norm_energy + criticality_term + duration_fairness - wait_boost + work_term
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score
