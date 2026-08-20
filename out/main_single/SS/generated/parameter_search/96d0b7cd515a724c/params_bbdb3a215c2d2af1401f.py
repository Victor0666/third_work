import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule: all numeric constants declared; only -2,-1,0,1,2 used inline."""
    eps = 0.01975273215075272
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
        x = np.abs(x)
        center = np.median(x)
        scale = np.median(np.abs(x - center)) + eps
        return (x - center) / scale
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    slack_penalty_base = np.where(norm_slack < 0, -np.power(np.clip(-norm_slack, 0.0, 2.1168913493896535), 2.4546835526354958), 0.0)
    slack_pressure_ratio = np.clip(-norm_slack, 0.0, 2.0)
    rank_boost = 0.8329668370341394 * norm_rank * (1.0 - np.exp(-0.41077878022037045 * slack_pressure_ratio))
    unc_gate_active = (norm_uncert > 0.4420438661995749) & (norm_slack < 0)
    duration_uncert_penalty = np.where(unc_gate_active, 1.6842762904586155 * norm_duration * np.abs(norm_slack), 0.0)
    wait_benefit = 0.4970929298685162 * (1.0 - np.exp(-norm_wait / (0.4970929298685162 + eps)))
    score = slack_penalty_base - rank_boost + 1.8047464217795663 * norm_energy + duration_uncert_penalty - wait_benefit
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    return score
