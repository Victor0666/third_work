import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: replaces fragile binary gate with smooth sigmoid DDL urgency gate,
       adds rank_boost_floor to prevent critical-path suppression under mild pressure,
       reverts to pure raw-slack-penalty dominance for violation enforcement,
       and simplifies interactions to improve robustness and CMA-ES convergence.
       All 12 parameters are used; no numeric literals beyond -2,-1,0,1,2."""
    eps = 0.021958881456129786
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
    ddl_urgency_gate = 1.0 / (1.0 + np.exp(-3.44414269881916 * urgency_input))
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.4044087741554043
    rank_boost = 0.041801372738953214 + (1.0 - 0.041801372738953214) * ddl_urgency_gate
    protected_rank = norm_rank * rank_boost
    wait_benefit = np.clip(0.36233549406012017 * (ready_wait_time + 2.680168533010686e-05), 0.0, 1.0)
    energy_uncert_penalty = norm_energy * norm_uncert * np.where((norm_energy > 0.0) & (norm_uncert > 0.5118882690520336), 1.0, 0.0)
    duration_risk_score = norm_duration * np.where((slack <= 0.021958881456129786) & (uncertainty >= 0.5118882690520336), 1.0, 0.0)
    score = +raw_slack_penalty - 1.7174722178467294 * protected_rank - 1.9941532705245189 * norm_energy - norm_duration - wait_benefit + 0.29967860197188306 * duration_risk_score + 0.25339365325754065 * energy_uncert_penalty + 0.8942962331508049 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
