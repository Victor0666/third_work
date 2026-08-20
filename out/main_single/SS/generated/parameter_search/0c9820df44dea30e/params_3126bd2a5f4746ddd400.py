import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: replaces fragile binary gate with smooth sigmoid DDL urgency gate,
       adds rank_boost_floor to prevent critical-path suppression under mild pressure,
       reverts to pure raw-slack-penalty dominance for violation enforcement,
       and simplifies interactions to improve robustness and CMA-ES convergence.
       All 12 parameters are used; no numeric literals beyond -2,-1,0,1,2."""
    eps = 0.008718460926703257
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
        center = np.median(x) if N > 1 else x[0]
        spread = np.median(np.abs(x - center)) if N > 1 else np.abs(x[0] - center) + eps
        return (x - center) / (spread + eps)
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    urgency_input = -slack / (np.abs(np.median(slack)) + eps) if N > 1 else -slack[0] / (np.abs(slack[0]) + eps)
    ddl_urgency_gate = 1.0 / (1.0 + np.exp(-3.099811404029862 * urgency_input))
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.858712454396312
    rank_boost = 0.14657270365854197 + (1.0 - 0.14657270365854197) * ddl_urgency_gate
    protected_rank = norm_rank * rank_boost
    wait_benefit = np.clip(0.0601743105943585 * (ready_wait_time + 3.998591582257758e-07), 0.0, 1.0)
    energy_uncert_penalty = norm_energy * norm_uncert * np.where((norm_energy > 0.0) & (norm_uncert > 0.9858135497242563), 1.0, 0.0)
    duration_risk_score = norm_duration * np.where((slack <= 0.008718460926703257) & (uncertainty >= 0.9858135497242563), 1.0, 0.0)
    score = +raw_slack_penalty - 1.2582516264908092 * protected_rank - 1.9450352876422414 * norm_energy - norm_duration - wait_benefit + 0.3025280645823707 * duration_risk_score + 0.1639763956396328 * energy_uncert_penalty + 1.9608940759282312 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
