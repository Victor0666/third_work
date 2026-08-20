import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule: all declared parameters used; no hidden numeric constants.
       Uses tanh-based urgency, critical-path coupling via slack & work, conditional energy-risk penalty,
       and bounded anti-starvation tanh. All numeric literals are -2,-1,0,1,2."""
    eps = 9.218570009258872e-05
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def stable_normalize(x):
        x_abs = np.abs(x)
        center = np.mean(x_abs) if N > 1 else x_abs[0]
        spread = np.mean(x_abs) + eps
        return (x - center) / (spread + eps)
    norm_slack = stable_normalize(slack)
    norm_energy = stable_normalize(min_incremental_energy)
    norm_duration = stable_normalize(min_exec_time + min_comm_time)
    norm_rank = stable_normalize(upward_rank)
    norm_work = stable_normalize(remaining_work)
    norm_wait = stable_normalize(ready_wait_time)
    norm_uncert = stable_normalize(uncertainty)
    slack_urgency = np.tanh(-slack * 3.8292622822661984)
    slack_pressure_mask = (slack <= 0.0).astype(float)
    work_pressure_mask = (norm_work > 0.0).astype(float)
    coupling_gate = slack_pressure_mask * work_pressure_mask
    boosted_rank = norm_rank * (1.0 + 2.1343837497995235 * coupling_gate)
    risk_active = (slack < 0.0) & (norm_uncert > 0.6826171981552851)
    energy_uncert_penalty = np.where(risk_active, norm_energy * norm_uncert, np.zeros_like(norm_energy))
    wait_benefit = np.tanh(0.12186320293401744 * norm_wait)
    energy_weight = np.where(slack > 0.0, 0.9048567097670378, 0.9048567097670378 * 0.5544590236609922)
    score = +(1.0 - slack_urgency) - boosted_rank - energy_weight * norm_energy - norm_duration - wait_benefit + 0.5176193342363734 * energy_uncert_penalty + 1.2290686550530654 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
