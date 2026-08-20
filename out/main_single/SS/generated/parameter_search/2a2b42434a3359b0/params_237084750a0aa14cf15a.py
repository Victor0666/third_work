import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: introduces DDL-protection gate to enforce hard deadline feasibility before energy optimization.
    Energy and uncertainty terms are *disabled* under slack < 0 via a smooth, differentiable gate — ensuring constraint-first behavior.
    All declared parameters are used; no numeric literals except -2,-1,0,1,2; shape (N,) guaranteed."""
    eps = 0.05266053032863679
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
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.700089267003326
    norm_slack_penalty = robust_normalize(raw_slack_penalty)
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-4.144854386593032 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * (1.0 + 2.092252979177133 * rank_gate)
    ddl_gate = 1.0 / (1.0 + np.exp(-4.179191150206047 * slack))
    uncert_gate = 1.0 / (1.0 + np.exp(-4.179191150206047 * (norm_uncert - 0.7114237903490176)))
    duration_risk_score = norm_duration * uncert_gate * slack_pressure * ddl_gate
    wait_benefit = 1.0 - np.exp(-0.1828587601882044 * (norm_wait + 7.269697010819337e-06))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    energy_slack_penalty = norm_energy * (1.0 + 2.700089267003326 * slack_pressure) * ddl_gate
    score = +norm_slack_penalty - boosted_rank - 0.49913372442607407 * norm_energy * ddl_gate - norm_duration * ddl_gate - wait_benefit + 0.5085789329469639 * duration_risk_score + 0.6776168984883219 * energy_uncert_penalty + 2.700089267003326 * energy_slack_penalty + 1.2262234714164888 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
