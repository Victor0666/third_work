import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: inherits robust sign-preserving normalization and unified slack pressure from Parent 2;
       adds energy-uncertainty interaction and slack-gated work leverage from Parent 1;
       removes harmful work_pressure mask; uses linear slack reward + power-law penalty for balanced feasibility-first ranking;
       all literals are -2,-1,0,1,2; epsilon via PARAMS; no side effects."""
    eps = 0.01603053075625464
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
        center = np.median(x)
        scale = np.median(np.abs(x - center)) + eps
        return (x - center) / scale
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    slack_pressure = np.power(np.clip(-norm_slack, 0.0, None), 2.1168635116054073)
    rank_gate = np.clip(1.0 - 2.0 * np.clip(norm_slack, 0.0, 0.44839869479892036), 0.0, 1.0)
    boosted_rank = norm_rank * (1.0 + 1.6250682663875482 * rank_gate)
    uncert_gate = ((norm_uncert > 0.2000613880180416) & (slack_pressure > 0)).astype(float)
    duration_risk_interaction = norm_duration * uncert_gate * 1.7472006321415168
    energy_uncert_gate = ((norm_uncert > 0.2000613880180416) & (slack_pressure > 0)).astype(float)
    energy_uncert_penalty = norm_energy * norm_uncert * energy_uncert_gate * 0.3543297843772406
    wait_benefit = np.tanh(0.4415579661414715 * norm_wait)
    slack_reward = 0.866300511990718 * np.clip(norm_slack, None, 0.0)
    work_leverage_gate = (slack >= 0.0).astype(float)
    work_leverage = norm_work * work_leverage_gate * 1.6034830841338872
    score = +slack_pressure + 0.7393451219223491 * norm_energy - boosted_rank + duration_risk_interaction - wait_benefit - slack_reward + energy_uncert_penalty - work_leverage
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
