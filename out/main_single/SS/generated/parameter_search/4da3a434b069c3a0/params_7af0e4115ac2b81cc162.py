import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: replaces MAD normalization with bounded min-max scaling
       to preserve deadline-critical slack/uncertainty signal fidelity; reintroduces
       energy_sensitivity as core linear term; removes redundant work/energy coupling;
       uses raw slack for pressure computation (not normalized) to avoid distortion
       under hard DDL constraints; all operations protected against NaN/inf."""
    eps = 0.00022517448839990222
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def bounded_minmax(x):
        x_min = np.min(x)
        x_max = np.max(x)
        range_val = x_max - x_min + eps
        return (x - x_min) / range_val
    norm_energy = bounded_minmax(min_incremental_energy)
    norm_duration = bounded_minmax(min_exec_time + min_comm_time)
    norm_rank = bounded_minmax(upward_rank)
    norm_wait = bounded_minmax(ready_wait_time)
    norm_uncert = bounded_minmax(uncertainty)
    slack_pressure_raw = np.clip(-slack, 0.0, None)
    slack_pressure = np.power(slack_pressure_raw + eps, 1.8142706482012163)
    norm_slack_for_gate = bounded_minmax(slack)
    rank_gate = np.clip(1.0 - 2.0 * np.clip(norm_slack_for_gate, 0.0, 0.37766969159916275), 0.0, 1.0)
    boosted_rank = norm_rank * (1.0 + 0.8104486414292695 * rank_gate)
    joint_risk_gate = ((norm_uncert > 0.9995295664715212) & (slack_pressure > 0)).astype(float)
    duration_risk_interaction = norm_duration * joint_risk_gate * 0.15786415592191572
    wait_benefit = np.tanh(0.5830926327669279 * norm_wait)
    coupled_rank = np.clip(norm_rank * (1.0 + 0.33403037603866675 * slack_pressure), 0.0, 2.0)
    score = +slack_pressure + 1.4202145343025883 * norm_energy - coupled_rank - wait_benefit + duration_risk_interaction
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    return score
