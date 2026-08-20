import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: replaces dual-gated urgency with *relative-slack* tanh urgency
       (tanh(-slack / (mean|slack| + eps))) for improved gradient sensitivity near deadlines;
       reverts successor-release to additive local-gated term (no coupling); removes global ddl_protection_boost
       to avoid diluting per-task urgency; retains robust mean-abs normalization and bounded tanh anti-starvation.
    """
    eps = 0.0017523982855486562
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def robust_norm(x):
        x_abs = np.abs(x)
        denom = np.mean(x_abs) + eps
        return x / denom
    norm_slack = robust_norm(slack)
    norm_energy = robust_norm(min_incremental_energy)
    norm_duration = robust_norm(min_exec_time + min_comm_time)
    norm_rank = robust_norm(upward_rank)
    norm_work = robust_norm(remaining_work)
    norm_wait = robust_norm(ready_wait_time)
    norm_uncert = robust_norm(uncertainty)
    slack_abs_mean = np.mean(np.abs(slack)) + eps
    relative_urgency = -slack / slack_abs_mean
    slack_urgency = np.tanh(2.947318201003219 * relative_urgency)
    norm_urgency = robust_norm(slack_urgency)
    slack_tight = (slack < 0.0) | (norm_slack < -0.28194832093315036)
    uncert_low = norm_uncert < 0.32062313237121093
    local_ddl_gate = np.where(slack_tight & uncert_low, 1.0, 0.0)
    boosted_rank = norm_rank * (1.0 + 1.3825070700826683 * local_ddl_gate)
    release_bonus = 1.2123182757795747 * norm_work * norm_rank * local_ddl_gate
    wait_benefit = np.tanh(0.36067487855533414 * (norm_wait + 3.428326975790413e-05))
    risk_energy_gate = np.where((slack < 0.0) & (norm_uncert > 0.32062313237121093), 1.0, 0.0)
    energy_uncert_penalty = norm_energy * risk_energy_gate * 0.9142832773714675
    score = +norm_urgency - boosted_rank - 1.4277992049515695 * norm_energy - wait_benefit - release_bonus + energy_uncert_penalty + 0.9611262014433419 * norm_work
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    assert score.shape == (N,), f'Expected shape {(N,)}, got {score.shape}'
    return score.reshape(-1)
