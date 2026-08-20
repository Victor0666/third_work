import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule with all numeric thresholds declared as parameters."""
    eps = 1.9685560139495037e-07

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
    slack_urgency = 2.9779488751507035 * np.tanh(np.clip(-norm_slack, 0, 1.9116098824658674)) + 0.712026163897793 * np.tanh(np.clip(norm_slack, 0, 2.1072779656641973))
    slack_pressure = np.clip(-norm_slack, 0, 1.9116098824658674)
    rank_gate = 1.0 / (1.0 + np.exp(0.0003785412937491227 * (slack_pressure - 2)))
    criticality_term = 1.5417983927827588 * rank_gate * norm_rank
    coupling = np.tanh(1.35562881126792 * norm_uncert * np.clip(-norm_slack, 0, 1.9116098824658674))
    duration_fairness = 0.1811201406727458 * np.abs(norm_duration)
    wait_boost = 0.14604269596189517 * norm_wait
    work_term = -0.42041106698670316 * norm_work
    score = slack_urgency + coupling + 1.5768859735297063 * norm_energy + criticality_term + duration_fairness - wait_boost + work_term
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score
