import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule: uses robust quantile-based normalization;
       introduces successor-release interaction (min_exec_time × clipped_upward_rank × (slack <= 0));
       enforces hard DDL feasibility via binary gating;
       retains only validated structural terms;
       uses sign-preserving linear slack urgency and bounded wait bonus."""
    eps = 9.667015548076938e-06
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def robust_quantile_normalize(x):
        x = np.copy(x)
        if N == 1:
            q_low = q_high = x[0]
        else:
            q_low = np.quantile(x, 0.22590583436586892)
            q_high = np.quantile(x, 0.691874653646479)
        iqr = q_high - q_low
        spread = iqr if iqr > eps else eps
        return (x - q_low) / spread
    norm_slack = robust_quantile_normalize(slack)
    norm_energy = robust_quantile_normalize(min_incremental_energy)
    norm_exec = robust_quantile_normalize(min_exec_time)
    norm_rank = robust_quantile_normalize(upward_rank)
    norm_work = robust_quantile_normalize(remaining_work)
    norm_wait = robust_quantile_normalize(ready_wait_time)
    norm_uncert = robust_quantile_normalize(uncertainty)
    ddl_breach = (slack <= 0.0).astype(float)
    clipped_rank = np.clip(norm_rank, 0.0, 2.0)
    successor_release = norm_exec * clipped_rank * ddl_breach
    critical_path_importance = norm_rank * norm_work * ddl_breach
    raw_slack_urgency = -np.clip(slack, -np.inf, 0.0)
    norm_slack_urgency = robust_quantile_normalize(raw_slack_urgency)
    slack_penalty = np.clip(norm_slack_urgency, -2.0, 2.0)
    wait_threshold = np.quantile(ready_wait_time, 0.691874653646479)
    uncert_threshold = np.quantile(uncertainty, 0.691874653646479)
    wait_high = (ready_wait_time >= wait_threshold).astype(float)
    uncert_high = (uncertainty >= uncert_threshold).astype(float)
    congestion_gate = wait_high * uncert_high
    energy_feasible = (slack >= 0.0).astype(float)
    energy_penalty = norm_energy * energy_feasible
    energy_uncert_coupling = norm_energy * norm_uncert * congestion_gate * energy_feasible
    wait_bonus = np.clip(ready_wait_time / (ready_wait_time + 4.5040282351218316e-05), 0.0, 1.0)
    score = +slack_penalty - 1.826824012694713 * critical_path_importance - 1.826824012694713 * successor_release - 0.32503726279686107 * energy_penalty - 0.6552355162462141 * energy_uncert_coupling - wait_bonus + 0.0030965315960723066 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
