import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule with IQR-based robust normalization, tanh urgency, 
    and explicit energy-uncertainty interaction — all parameters declared."""
    eps = 3.3952146334836714e-05
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def robust_normalize(x):
        x = np.asarray(x, dtype=float)
        q_low = np.quantile(x, 0.3936706248615467)
        q_high = np.quantile(x, 0.7689473645453393)
        iqr = q_high - q_low + eps
        center = np.median(x)
        return (x - center) / iqr
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    slack_urgency = 14.977220797689597 * np.tanh(np.clip(-norm_slack, 0, 2)) + 0.8895952769018011 * np.tanh(np.clip(norm_slack, 0, 1))
    slack_pressure = np.clip(-norm_slack, 0, 2)
    rank_gate = 1.0 / (1.0 + np.exp(0.021113958073869024 * (slack_pressure - 2)))
    criticality_term = 0.7861423015464949 * rank_gate * norm_rank
    coupling = np.tanh(0.028025823553488517 * norm_uncert * np.clip(-norm_slack, 0, 2))
    duration_fairness = 1.6629350942785872 * np.abs(norm_duration)
    wait_boost = 0.7190137151511368 * norm_wait
    work_term = -0.24737407950596552 * norm_work
    energy_term = 3.03410284731583 * norm_energy + 1.0 * norm_energy * norm_uncert
    score = slack_urgency + coupling + energy_term + criticality_term + duration_fairness - wait_boost + work_term
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
