import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: inherits robust sign-preserving normalization and unified slack pressure from Parent 2;
       adds energy-uncertainty interaction and slack-gated work leverage from Parent 1;
       removes harmful work_pressure mask; uses linear slack reward + power-law penalty for balanced feasibility-first ranking;
       all literals are -2,-1,0,1,2; epsilon via PARAMS; no side effects."""
    eps = 0.013204024739424098
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
    slack_pressure = np.power(np.clip(-norm_slack, 0.0, None), 2.0734145720644914)
    rank_gate = np.clip(1.0 - 2.0 * np.clip(norm_slack, 0.0, 0.4078336650832606), 0.0, 1.0)
    boosted_rank = norm_rank * (1.0 + 1.4865610433246979 * rank_gate)
    uncert_gate = ((norm_uncert > 0.16141368571950482) & (slack_pressure > 0)).astype(float)
    duration_risk_interaction = norm_duration * uncert_gate * 1.4527950150664357
    energy_uncert_gate = ((norm_uncert > 0.16141368571950482) & (slack_pressure > 0)).astype(float)
    energy_uncert_penalty = norm_energy * norm_uncert * energy_uncert_gate * 0.4080482662995255
    wait_benefit = np.tanh(0.5168662990017602 * norm_wait)
    slack_reward = 1.5137473744566337 * np.clip(norm_slack, None, 0.0)
    work_leverage_gate = (slack >= 0.0).astype(float)
    work_leverage = norm_work * work_leverage_gate * 0.7031694955697907
    score = +slack_pressure + 0.33161286018016206 * norm_energy - boosted_rank + duration_risk_interaction - wait_benefit - slack_reward + energy_uncert_penalty - work_leverage
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
