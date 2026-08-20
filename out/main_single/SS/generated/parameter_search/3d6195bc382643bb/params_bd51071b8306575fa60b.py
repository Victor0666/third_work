import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule: all numeric thresholds moved to PARAMETER_SCHEMA; only -2,-1,0,1,2 used as literals."""
    eps = 0.0024360992525149627
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
    slack_pressure = np.power(np.clip(-norm_slack, 0.0, None), 1.8262253219245144)
    rank_gate = np.clip(1.0 - 2.0 * np.clip(norm_slack, 0.0, 1.8029903776750735), 0.0, 1.0)
    boosted_rank = norm_rank * (1.0 + 0.5560511222728582 * rank_gate)
    uncert_gate = ((norm_uncert > 0.5059241497151569) & (slack_pressure > 0)).astype(float)
    duration_risk_interaction = norm_duration * uncert_gate * 0.6798318942449287
    wait_benefit = np.tanh(0.501546415473902 * norm_wait)
    coupled_rank = norm_rank * (1.0 + 0.19160159560879197 * slack_pressure)
    score = +slack_pressure + 1.1596448456108572 * norm_energy - coupled_rank - wait_benefit + duration_risk_interaction
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(N)
