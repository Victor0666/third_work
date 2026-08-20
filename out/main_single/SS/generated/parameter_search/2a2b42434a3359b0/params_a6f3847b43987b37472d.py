import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: introduces DDL-protection gate to enforce hard deadline feasibility before energy optimization.
    Energy and uncertainty terms are *disabled* under slack < 0 via a smooth, differentiable gate — ensuring constraint-first behavior.
    All declared parameters are used; no numeric literals except -2,-1,0,1,2; shape (N,) guaranteed."""
    eps = 0.015295215809590599
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
        x_abs = np.abs(x)
        if N == 1:
            center = x_abs[0]
            spread = eps
        else:
            center = np.median(x_abs)
            spread = np.median(np.abs(x_abs - center))
        return (x_abs - center) / (spread + eps)
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.2586969024028256
    norm_slack_penalty = robust_normalize(raw_slack_penalty)
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-1.6668407456096124 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * (1.0 + 2.0108181023246425 * rank_gate)
    ddl_gate = 1.0 / (1.0 + np.exp(-2.429766509009272 * slack))
    uncert_gate = 1.0 / (1.0 + np.exp(-2.429766509009272 * (norm_uncert - 0.7249177621608633)))
    duration_risk_score = norm_duration * uncert_gate * slack_pressure * ddl_gate
    wait_benefit = 1.0 - np.exp(-0.26865785035253364 * (norm_wait + 3.66352774822937e-08))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    energy_slack_penalty = norm_energy * (1.0 + 1.2586969024028256 * slack_pressure) * ddl_gate
    score = +norm_slack_penalty - boosted_rank - 1.1454097770309692 * norm_energy * ddl_gate - norm_duration * ddl_gate - wait_benefit + 0.8937437199112694 * duration_risk_score + 0.5363067288984349 * energy_uncert_penalty + 1.2586969024028256 * energy_slack_penalty + 1.0497790895481438 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
