import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: replaces static duration penalty with uncertainty-weighted duration robustness;
       retains original clipped linear DDL gate (0→1 over [0,1]) for feasibility-first enforcement;
       reverts to linear successor-release interaction (norm_rank * norm_work * ddl_pressure) per consensus validation;
       adds explicit robustness term: penalizes tasks whose total duration (exec+comm) has high uncertainty *and* is large;
       all clipping bounds preserved at [-2,2] for AST depth control and numerical stability."""
    eps = 9.081917237111887e-05
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def median_mad_normalize(x):
        x = np.copy(x)
        if N == 1:
            med = x[0]
            mad = eps
        else:
            med = np.median(x)
            mad = np.median(np.abs(x - med))
        spread = mad if mad > eps else eps
        return (x - med) / spread
    norm_slack = median_mad_normalize(slack)
    norm_energy = median_mad_normalize(min_incremental_energy)
    total_duration = min_exec_time + min_comm_time
    norm_duration = median_mad_normalize(total_duration)
    norm_rank = median_mad_normalize(upward_rank)
    norm_work = median_mad_normalize(remaining_work)
    norm_wait = median_mad_normalize(ready_wait_time)
    norm_uncert = median_mad_normalize(uncertainty)
    ddl_gate = np.clip(slack, 0.0, 1.0)
    ddl_gate = np.where(slack < 0.0, 0.0, ddl_gate)
    ddl_pressure = np.clip(-norm_slack, 0.0, 1.0)
    successor_release = norm_rank * norm_work * ddl_pressure
    raw_slack_penalty = np.maximum(-slack, 0.0)
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    coupled_rank = norm_rank * (1.0 + 2.1406096514717063 * ddl_pressure)
    uncert_gate = np.where(norm_uncert > 0.4508427628824517, 1.0, 0.0)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    wait_benefit = np.clip(0.14867347734459307 * ready_wait_time, 0.0, 1.0)
    robust_duration_penalty = norm_duration * norm_uncert * ddl_pressure * 0.6805057700972869
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - np.clip(coupled_rank, -2.0, 2.0) - 1.5006227613021719 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(robust_duration_penalty, -2.0, 2.0) + 0.154188933074818 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 0.8562608909301368 * np.clip(norm_work, -2.0, 2.0) + np.clip(ddl_pressure * 6.273841389943951, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
