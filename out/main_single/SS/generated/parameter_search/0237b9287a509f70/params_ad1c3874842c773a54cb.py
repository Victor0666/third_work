import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: sign-preserving robust normalization fixes urgency/starvation distortion;
    reintroduces linear slack term to promote early scheduling of safe tasks (reducing deadline violations);
    removes harmful work_pressure; uses unified slack_pressure only for penalties, not rewards;
    all literals are -2,-1,0,1,2; epsilon via PARAMS; no side effects."""
    eps = 1.2332057463917813e-06
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
    slack_pressure = np.power(np.clip(-norm_slack, 0.0, None), 1.003198802102946)
    rank_gate = np.clip(1.0 - 2.0 * np.clip(norm_slack, 0.0, 0.5027460462670073), 0.0, 1.0)
    boosted_rank = norm_rank * (1.0 + 2.203163911578722 * rank_gate)
    uncert_gate = ((norm_uncert > 0.6272392446710511) & (slack_pressure > 0)).astype(float)
    duration_risk_interaction = norm_duration * uncert_gate * 1.4571694486764695
    wait_benefit = np.tanh(0.7004285211368393 * norm_wait)
    slack_reward = 1.7681384684723465 * np.clip(norm_slack, None, 0.0)
    score = +slack_pressure + 0.5459193746017706 * norm_energy - boosted_rank + duration_risk_interaction - wait_benefit - slack_reward
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    return score
