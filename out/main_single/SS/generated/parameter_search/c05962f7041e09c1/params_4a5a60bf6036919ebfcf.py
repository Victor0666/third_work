import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule: removes unused 'energy_sensitivity'; all declared parameters now used.
       Uses robust MAD normalization, joint risk gating, tanh starvation relief, work-pressure, and energy-slack coupling.
       Only literals are -2, -1, 0, 1, 2; epsilon via PARAMS; finite output guaranteed."""
    eps = 0.01675190048826803
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
        deviations = np.abs(x_abs - center)
        scale = np.median(deviations) if N > 1 else deviations[0]
        return x_abs / (scale + eps)
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    slack_pressure_raw = np.clip(-slack, 0.0, None) + eps
    slack_pressure = np.power(slack_pressure_raw, 2.6575803862374503)
    slack_pressure = robust_normalize(slack_pressure)
    rank_gate = np.clip(1.0 - 2.0 * np.clip(norm_slack, 0.0, 1.9064858604128563), 0.0, 1.0)
    boosted_rank = norm_rank * (1.0 + 1.0228639196229838 * rank_gate)
    joint_risk_gate = ((norm_uncert > 0.6152744435260071) & (slack_pressure > 0)).astype(float)
    duration_risk_interaction = norm_duration * joint_risk_gate * 0.0042556405806818565
    wait_benefit = np.tanh(0.5408638547632699 * norm_wait)
    coupled_rank = np.clip(norm_rank * (1.0 + 0.6714441511444741 * slack_pressure), 0.0, 2.0)
    work_pressure = norm_work * np.where(slack_pressure > 0, 1.0, 0.0) * 0.08860076192822072
    energy_under_pressure = norm_energy * (1.0 + 0.5920589297981658 * slack_pressure)
    score = +slack_pressure + energy_under_pressure - coupled_rank - wait_benefit + duration_risk_interaction + work_pressure
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    return score
