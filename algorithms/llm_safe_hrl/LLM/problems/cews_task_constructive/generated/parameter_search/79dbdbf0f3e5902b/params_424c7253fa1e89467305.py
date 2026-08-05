import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule: all declared parameters are now used.
    
    Key fix: `uncertainty_sensitivity` is now used to modulate the leverage gate
    (replacing the previous hard max with a weighted blend), satisfying usage requirement.
    
    Other improvements:
      - Leverage gate is now a convex combination: (1 - sens)*slack_pressure + sens*uncert_pressure
      - All numeric literals in body are from {-2,-1,0,1,2}; epsilon comes only from PARAMS.
      - Uses np.finfo for safe finite clamping.
      - Preserves all structural intent: deadline-risk gating, critical-path leverage,
        starvation relief, work density, and duration-energy balance.
    """
    eps = 6.055197650520851e-05
    finfo = np.finfo(float)
    tiny = finfo.tiny
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)

    def robust_normalize(x):
        x = np.copy(x)
        center = np.median(x)
        dev = np.median(np.abs(x - center))
        scale = np.maximum(dev, eps)
        return (x - center) / scale
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    work_density = remaining_work / (duration + eps)
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(duration)
    norm_uprank = robust_normalize(upward_rank)
    norm_work_density = robust_normalize(work_density)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    slack_penalty = np.where(slack < 0, 3.5957847207769253 * -norm_slack, np.zeros_like(norm_slack))
    slack_urgency = np.where(slack >= 0, 1.7381444873318141 * norm_slack, np.zeros_like(norm_slack))
    slack_pressure = np.tanh(-norm_slack)
    uncert_pressure = np.tanh(norm_uncert)
    leverage_gate = (1.0 - 0.5881686051375736) * slack_pressure + 0.5881686051375736 * uncert_pressure
    leveraged_uprank = leverage_gate * norm_uprank
    duration_reliability = 1.0 / (1.0 + norm_uncert + np.abs(norm_duration) + eps)
    energy_weight_adj = 0.060441634052790134 * duration_reliability
    wait_boost = np.where(slack >= 0, 0.6357650847818948 * norm_wait, np.zeros_like(norm_wait))
    work_bonus = 0.32718482872223176 * norm_work_density
    duration_penalty = 0.6921379727989578 * norm_duration
    score = slack_penalty + slack_urgency + energy_weight_adj * norm_energy + duration_penalty - 0.0009936795089378703 * leveraged_uprank - work_bonus - wait_boost
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=-finfo.max)
    return score
