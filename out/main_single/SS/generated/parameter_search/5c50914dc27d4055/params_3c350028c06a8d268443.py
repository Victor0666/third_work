import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule: 12 parameters, all used; no unused 'successor_release_weight'."""
    eps = 0.00015096153896886378
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
    ddl_protection_gate = 1.0 / (1.0 + np.exp(-1.5910633006717683 * (slack_distance + eps)))
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.8335888358805723
    norm_slack_penalty = robust_normalize(raw_slack_penalty)
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-5.4696095698871705 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * (1.0 + 1.1911955087305628 * rank_gate)
    uncert_gate = np.where(norm_uncert > 0.19041489379811102, 1.0, 0.0)
    duration_risk_score = norm_duration * uncert_gate * ddl_protection_gate
    wait_benefit = 1.0 - np.exp(-0.598197819806207 * (norm_wait + 0.00020899209662086568))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate
    score = +norm_slack_penalty - boosted_rank - 0.9551129151424025 * norm_energy - norm_duration - wait_benefit + 0.9380333649617816 * duration_risk_score + 0.5317384560172276 * energy_uncert_penalty + 1.0753623375236465 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
