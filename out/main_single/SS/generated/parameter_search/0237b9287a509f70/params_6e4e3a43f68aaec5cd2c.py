import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: sign-preserving robust normalization fixes urgency/starvation distortion;
    reintroduces linear slack term to promote early scheduling of safe tasks (reducing deadline violations);
    removes harmful work_pressure; uses unified slack_pressure only for penalties, not rewards;
    all literals are -2,-1,0,1,2; epsilon via PARAMS; no side effects."""
    eps = 1.6260315322798495e-05
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
    slack_pressure = np.power(np.clip(-norm_slack, 0.0, None), 1.0440110376288763)
    rank_gate = np.clip(1.0 - 2.0 * np.clip(norm_slack, 0.0, 0.4184698121600349), 0.0, 1.0)
    boosted_rank = norm_rank * (1.0 + 1.712749519859985 * rank_gate)
    uncert_gate = ((norm_uncert > 0.7084008957674636) & (slack_pressure > 0)).astype(float)
    duration_risk_interaction = norm_duration * uncert_gate * 0.8681581172038418
    wait_benefit = np.tanh(0.4330483603330343 * norm_wait)
    slack_reward = 1.6052261320731667 * np.clip(norm_slack, None, 0.0)
    score = +slack_pressure + 0.31358046164213416 * norm_energy - boosted_rank + duration_risk_interaction - wait_benefit - slack_reward
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    return score
