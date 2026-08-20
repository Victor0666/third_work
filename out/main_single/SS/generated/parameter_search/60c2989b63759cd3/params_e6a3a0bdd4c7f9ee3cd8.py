import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: replaces dual-gated urgency with *relative-slack* tanh urgency
       (tanh(-slack / (mean|slack| + eps))) for improved gradient sensitivity near deadlines;
       reverts successor-release to additive local-gated term (no coupling); removes global ddl_protection_boost
       to avoid diluting per-task urgency; retains robust mean-abs normalization and bounded tanh anti-starvation.
    """
    eps = 0.01659125911433786
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
    slack_urgency = np.tanh(4.412223294200682 * relative_urgency)
    norm_urgency = robust_norm(slack_urgency)
    slack_tight = (slack < 0.0) | (norm_slack < -0.369035205968262)
    uncert_low = norm_uncert < 0.1303237502284196
    local_ddl_gate = np.where(slack_tight & uncert_low, 1.0, 0.0)
    boosted_rank = norm_rank * (1.0 + 0.5569925043147788 * local_ddl_gate)
    release_bonus = 0.9117926393309715 * norm_work * norm_rank * local_ddl_gate
    wait_benefit = np.tanh(0.06428864953233474 * (norm_wait + 8.791967333282709e-06))
    risk_energy_gate = np.where((slack < 0.0) & (norm_uncert > 0.1303237502284196), 1.0, 0.0)
    energy_uncert_penalty = norm_energy * risk_energy_gate * 0.7042095163279206
    score = +norm_urgency - boosted_rank - 1.1231373855487696 * norm_energy - wait_benefit - release_bonus + energy_uncert_penalty + 1.0899810693685292 * norm_work
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    assert score.shape == (N,), f'Expected shape {(N,)}, got {score.shape}'
    return score.reshape(-1)
