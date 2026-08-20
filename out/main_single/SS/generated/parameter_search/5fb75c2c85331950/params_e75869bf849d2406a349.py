import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: removes aggregated duration penalty to restore discriminative power;
       introduces *exec_comm_separation_factor* to independently gate and scale execution vs communication penalties;
       preserves all successful normalization, gating, and slack-coupling from Parent 2;
       ensures compute-bound tasks get urgent scheduling under tight slack while I/O-bound tasks are deprioritized only if bandwidth-limited;
       avoids structural overloading by decoupling exec/comm signals instead of summing them."""
    eps = np.finfo(float).tiny
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
        x = np.copy(x)
        if N == 1:
            center = x[0]
            spread = eps
        else:
            center = np.mean(x)
            spread = np.std(x, ddof=0)
        spread = np.where(spread > eps, spread, eps)
        return (x - center) / spread
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_exec = robust_normalize(min_exec_time)
    norm_comm = robust_normalize(min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    ddl_gate = 1.0 / (1.0 + np.exp(-2.7791295447697997 * slack))
    raw_slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    slack_pressure_gate = 1.0 / (1.0 + np.exp(-3.499888053067301 * (raw_slack_pressure - 1.0)))
    ddl_breach = (slack <= 0.0).astype(float)
    critical_path_score = norm_rank * norm_work * ddl_breach
    rank_amplifier = 1.0 + 0.7197542041959626 * slack_pressure_gate
    rank_amplifier = np.clip(rank_amplifier, 0.26334672610367904, 1.5152877087717183)
    boosted_rank = norm_rank * rank_amplifier
    uncert_gate = 1.0 / (1.0 + np.exp(-2.7791295447697997 * (norm_uncert - 0.22249746357358385)))
    comm_uncert_penalty = norm_comm * norm_uncert * uncert_gate * ddl_breach
    wait_bonus = 0.5892903775176989 * norm_wait
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    energy_slack_penalty = norm_energy * raw_slack_pressure * ddl_gate
    exec_penalty = 0.38662784173312303 * norm_exec * slack_pressure_gate * ddl_gate
    comm_penalty = (1.0 - 0.38662784173312303) * norm_comm * slack_pressure_gate * ddl_gate
    score = +np.clip(norm_slack, -2.0, 2.0) + np.clip(raw_slack_pressure * 2.534554361381955, -2.0, 2.0) - np.clip(critical_path_score, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 0.22660831779053667 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) + np.clip(wait_bonus, -2.0, 2.0) + 0.33286911504848715 * np.clip(energy_uncert_penalty, -2.0, 2.0) + np.clip(comm_uncert_penalty, -2.0, 2.0) + 0.5250846776145135 * np.clip(norm_work, -2.0, 2.0) + np.clip(exec_penalty, -2.0, 2.0) + np.clip(comm_penalty, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
