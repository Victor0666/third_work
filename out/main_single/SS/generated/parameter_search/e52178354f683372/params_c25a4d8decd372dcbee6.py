import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule: adds conditional DDL protection gate; removes redundant successor_release_weight."""
    eps = 0.00039046424504638036
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
        center = np.median(x_abs) if N > 1 else x_abs[0]
        spread = np.median(np.abs(x_abs - center)) if N > 1 else np.abs(x_abs[0] - center) + eps
        return (x_abs - center) / (spread + eps)
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    slack_distance = -slack
    ddl_protection_gate = 1.0 / (1.0 + np.exp(-0.6459545936326979 * (slack_distance + eps)))
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.9394304713358532
    norm_slack_penalty = robust_normalize(raw_slack_penalty)
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-4.167839146827943 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * (1.0 + 1.437195173227432 * rank_gate)
    uncert_gate = np.where(norm_uncert > 0.3085059207142293, 1.0, 0.0)
    duration_risk_score = norm_duration * uncert_gate * ddl_protection_gate
    wait_benefit = 1.0 - np.exp(-0.3733354718976054 * (norm_wait + 3.9393802648689765e-06))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate
    score = +norm_slack_penalty - boosted_rank - 1.6723716271318454 * norm_energy - norm_duration - wait_benefit + 1.3446884528365728 * duration_risk_score + 0.3317690241764047 * energy_uncert_penalty + 0.5343907998915349 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
