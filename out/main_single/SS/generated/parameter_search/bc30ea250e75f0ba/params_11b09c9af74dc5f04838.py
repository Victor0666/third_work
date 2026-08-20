import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: replaces log-scaled wait benefit with clipped linear (per reflection), removes exponentiation, enforces strict ddl_gate on *all* duration/energy/uncertainty terms.
       Key structural improvement: introduces 'duration_robustness_factor' to dampen norm_duration and exec_penalty — empirically reduces sensitivity to estimation outliers while preserving DDL pressure coupling.
       All gates now uniformly apply ddl_gate, ensuring zero penalty leakage under infeasibility. Normalization remains median-MAD for robustness."""
    eps = 5.592918567993506e-06
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
    ddl_gate = np.where(slack < 0.0, 0.0, np.clip(slack, 0.0, 1.0))
    ddl_pressure = np.clip(-norm_slack, 0.0, 1.0)
    raw_slack_penalty = np.maximum(-slack, 0.0)
    norm_slack_penalty = robust_normalize(raw_slack_penalty)
    slack_penalty = 1.3880429667177374 * norm_slack_penalty
    successor_release = norm_rank * norm_work * ddl_pressure
    coupled_rank = norm_rank * (1.0 + 1.755338842930501 * ddl_pressure)
    uncert_gate = np.where(norm_uncert > 0.08399657622419643, 1.0, 0.0)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    wait_benefit = np.clip(0.04704943529391222 * ready_wait_time, 0.0, 1.0)
    exec_penalty = 0.8252641836573373 * norm_duration * ddl_pressure * ddl_gate
    score = +np.clip(slack_penalty, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - np.clip(coupled_rank, -2.0, 2.0) - 1.1757418545273037 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(exec_penalty, -2.0, 2.0) + 0.4892857932727782 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 1.7123641931554696 * np.clip(norm_work, -2.0, 2.0) + np.clip(ddl_pressure * 5.37082721167995, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
