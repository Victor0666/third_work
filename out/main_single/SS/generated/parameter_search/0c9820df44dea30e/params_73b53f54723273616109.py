import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: replaces fragile binary gate with smooth sigmoid DDL urgency gate,
       adds rank_boost_floor to prevent critical-path suppression under mild pressure,
       reverts to pure raw-slack-penalty dominance for violation enforcement,
       and simplifies interactions to improve robustness and CMA-ES convergence.
       All 12 parameters are used; no numeric literals beyond -2,-1,0,1,2."""
    eps = 0.09890535688939603
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
        center = np.median(x) if N > 1 else x[0]
        spread = np.median(np.abs(x - center)) if N > 1 else np.abs(x[0] - center) + eps
        return (x - center) / (spread + eps)
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    urgency_input = -slack / (np.abs(np.median(slack)) + eps) if N > 1 else -slack[0] / (np.abs(slack[0]) + eps)
    ddl_urgency_gate = 1.0 / (1.0 + np.exp(-2.167277525631384 * urgency_input))
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.005285490169424
    rank_boost = 0.18103457464445602 + (1.0 - 0.18103457464445602) * ddl_urgency_gate
    protected_rank = norm_rank * rank_boost
    wait_benefit = np.clip(0.009670183671085499 * (ready_wait_time + 1.2238330097438323e-07), 0.0, 1.0)
    energy_uncert_penalty = norm_energy * norm_uncert * np.where((norm_energy > 0.0) & (norm_uncert > 0.5256546006868005), 1.0, 0.0)
    duration_risk_score = norm_duration * np.where((slack <= 0.09890535688939603) & (uncertainty >= 0.5256546006868005), 1.0, 0.0)
    score = +raw_slack_penalty - 2.6818935117489264 * protected_rank - 1.3813157611501619 * norm_energy - norm_duration - wait_benefit + 0.5667903470348583 * duration_risk_score + 0.4841050429541326 * energy_uncert_penalty + 0.7006235716301482 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
