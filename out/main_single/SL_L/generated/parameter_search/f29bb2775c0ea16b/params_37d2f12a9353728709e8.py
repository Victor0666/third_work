import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """
    Hybrid priority rule with unified tanh-based slack pressure, joint slack+uncertainty gating
    for critical path leverage, saturating wait fairness, and robust MAD normalization.
    All parameters declared in PARAMETER_SCHEMA are used; no unused or missing references.
    Numeric literals restricted to {-2,-1,0,1,2}; epsilon guarded via PARAMS["epsilon"].
    """
    eps = 7.778871461750163e-09
    finfo = np.finfo(np.float64)
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    slack = np.nan_to_num(np.asarray(slack, dtype=np.float64), nan=0.0, posinf=finfo.max, neginf=finfo.min)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=np.float64), nan=0.0, posinf=finfo.max, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=eps)

    def mad_normalize(x):
        x = np.asarray(x, dtype=np.float64)
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        scale = 1.375544110516619 * (mad if mad > eps else eps)
        return (x - med) / scale
    slack_pressure = np.tanh((1 + 0.9996111631934274) * -slack)
    slack_norm = np.clip(1 + slack_pressure, 0.0, 2.0)
    duration = min_exec_time + min_comm_time
    duration_norm = mad_normalize(duration)
    energy_norm = mad_normalize(min_incremental_energy)
    energy_score = 1.767591569955356 * energy_norm
    rank_norm = mad_normalize(upward_rank)
    unc_med = np.median(uncertainty)
    unc_gate = np.clip(1.0 - (uncertainty - unc_med) / np.maximum(unc_med, eps), 0.0, 1.0)
    critical_gate = ((slack >= 0.0) * unc_gate).astype(float)
    critical_boost = 3.210167448330161 * rank_norm * critical_gate
    work_density = np.divide(remaining_work, duration + eps)
    work_density_norm = mad_normalize(work_density)
    work_density_gate = (slack >= 0.0).astype(float)
    work_density_bonus = 0.2618784848296387 * work_density_norm * work_density_gate
    wait_norm = mad_normalize(ready_wait_time)
    wait_boost = 1.0 - np.exp(-0.13862445467841067 * np.maximum(wait_norm, 0.0))
    wait_score = wait_boost * (slack >= 0.0).astype(float)
    unc_norm = mad_normalize(uncertainty)
    uncertainty_amplifier = 0.5080106142536228 * unc_norm * np.abs(slack_pressure)
    score = slack_norm + 1.00277512769094 * duration_norm + energy_score - critical_boost - work_density_bonus + wait_score + uncertainty_amplifier
    score = np.nan_to_num(score, nan=finfo.max, posinf=finfo.max, neginf=-finfo.max)
    return score
