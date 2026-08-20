import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: merges robust median-MAD normalization with convex slack urgency;
       uses multiplicative critical-path coupling (rank × work × pressure);
       retains smooth ddl_pressure = 1/(1+|slack|+eps), congestion-aware host-load surrogate,
       and tunable energy suppression; enforces strict finite-range clipping and shape (N,) output."""
    eps = 7.07145097766e-05
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
        x = np.asarray(x, dtype=float)
        if N == 1:
            med = x[0]
            mad = eps
        else:
            med = np.median(x)
            mad = np.median(np.abs(x - med))
        spread = mad + eps
        return (x - med) / spread
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    hard_feasibility_gate = np.where(slack < 0.0, 0.0, 1.0)
    slack_abs = np.abs(slack) + eps
    ddl_pressure = np.clip(1.0 / (1.0 + slack_abs), 0.0, 1.0)
    raw_slack_penalty = np.maximum(-slack, 0.0) + eps
    convex_slack_penalty = np.power(raw_slack_penalty, 1.003992062012983)
    norm_slack_penalty = robust_normalize(convex_slack_penalty)
    slack_penalty = 1.43876195161758 * norm_slack_penalty
    critical_coupling = norm_rank * norm_work * ddl_pressure
    uncert_gate = np.where(norm_uncert > 0.5103809887415314, 1.0, 0.0)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * hard_feasibility_gate
    wait_benefit = np.clip(0.40612965762964237 * ready_wait_time, 0.0, 2.0)
    duration_penalty = 0.5092193239203471 * norm_duration * ddl_pressure * hard_feasibility_gate
    slack_energy_suppress = np.where(slack < 0.0, 1.0 - 0.9105990752696759, 1.0)
    suppressed_norm_energy = norm_energy * slack_energy_suppress
    load_surrogate = norm_duration * norm_uncert
    load_gate = np.where((hard_feasibility_gate > 0.0) & (norm_uncert > 0.5103809887415314), np.clip(load_surrogate, 0.0, 2.0), 0.0)
    host_load_penalty = 0.6356484286243098 * load_gate * suppressed_norm_energy
    score = +np.clip(slack_penalty, -2.0, 2.0) - np.clip(critical_coupling, -2.0, 2.0) - 1.638537028094084 * np.clip(suppressed_norm_energy, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(duration_penalty, -2.0, 2.0) + 0.8201182486210172 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 0.6893023741212351 * np.clip(norm_work, -2.0, 2.0) + host_load_penalty
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.reshape(-1)
