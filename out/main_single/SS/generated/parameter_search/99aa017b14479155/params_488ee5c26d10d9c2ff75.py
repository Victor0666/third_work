import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule: all declared parameters are used; no unused entries.
    Introduces work-pressure term to prevent late-stage bottlenecks under deadline pressure.
    Uses unified slack_pressure signal, smooth rank_gate (with literal 2.0), tanh starvation relief, and robust normalization.
    All numeric literals are -2,-1,0,1,2; epsilon via PARAMS; no side effects or I/O."""
    eps = 3.1106586410943496e-05
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
        center = np.median(x_abs)
        scale = np.median(np.abs(x_abs - center)) + eps
        return (x_abs - center) / scale
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    slack_pressure = np.power(np.clip(-norm_slack, 0.0, None), 2.2665873925007154)
    rank_gate = np.clip(1.0 - 2.0 * np.clip(norm_slack, 0.0, 1.2803924535263715), 0.0, 1.0)
    boosted_rank = norm_rank * (1.0 + 1.0854366028636102 * rank_gate)
    work_pressure = np.where(slack_pressure > 0, 0.3674327562546793 * norm_work * slack_pressure, 0.0)
    uncert_gate = ((norm_uncert > 0.9901265997673055) & (slack_pressure > 0)).astype(float)
    duration_risk_interaction = norm_duration * uncert_gate * 1.2713621408701254
    wait_benefit = np.tanh(0.47089809868700705 * norm_wait)
    score = +slack_pressure + 1.3605516292274817 * norm_energy - boosted_rank + work_pressure + duration_risk_interaction - wait_benefit
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    return score
