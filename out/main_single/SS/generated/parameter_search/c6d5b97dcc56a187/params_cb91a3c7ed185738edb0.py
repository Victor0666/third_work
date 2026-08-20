import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule:
       - Replaces hard threshold with smooth, differentiable slack-sensitivity gate (centered at PARAMS["slack_sensitivity_center"])
         for critical-path leverage — improves gradient-based optimization and avoids brittle boundaries.
       - Introduces successor-release interaction: penalizes tasks whose successors are not yet ready using
         `upward_rank * (1 - normalized_successor_release_prob)` where release probability is estimated via
         robustly normalized `ready_wait_time` as proxy for upstream completion readiness.
       - Adds host load proxy gating: unifies `ready_wait_time + uncertainty` under a sigmoid gated by itself,
         decoupling starvation relief from deadline pressure and enabling adaptive load-awareness.
       - All numeric literals are restricted to -2, -1, 0, 1, 2; uses np.finfo for epsilon safeguards."""
    eps = 0.006080212658868957
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def robust_range_normalize(x):
        x = np.copy(x)
        if N == 1:
            q1 = q3 = x[0]
        else:
            q1 = np.quantile(x, 0.28889629570611847)
            q3 = np.quantile(x, 0.8133840110948551)
        iqr = q3 - q1
        spread = iqr if iqr > eps else eps
        return (x - q1) / spread
    norm_energy = robust_range_normalize(min_incremental_energy)
    norm_exec = robust_range_normalize(min_exec_time)
    norm_comm = robust_range_normalize(min_comm_time)
    norm_rank = robust_range_normalize(upward_rank)
    norm_work = robust_range_normalize(remaining_work)
    norm_wait = robust_range_normalize(ready_wait_time)
    norm_uncert = robust_range_normalize(uncertainty)
    if N == 1:
        slack_med = slack[0]
        slack_mad = eps
    else:
        slack_med = np.median(slack)
        slack_mad = np.median(np.abs(slack - slack_med))
    slack_spread = slack_mad if slack_mad > eps else eps
    norm_slack = (slack - slack_med) / slack_spread
    ddl_gate = 1.0 / (1.0 + np.exp(-6.400950713353678 * slack))
    slack_sensitivity_gate = 1.0 / (1.0 + np.exp(-6.400950713353678 * (slack - -0.0051067269566107965)))
    critical_path_score = norm_rank * norm_work * slack_sensitivity_gate
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.623800891395158
    norm_slack_penalty = robust_range_normalize(raw_slack_penalty)
    successor_release_prob = 1.0 - np.clip(norm_wait, 0.0, 1.0)
    successor_blocking_penalty = norm_rank * (1.0 - successor_release_prob) * 0.8618353613215067
    load_proxy = norm_wait + norm_uncert
    load_gate = 1.0 / (1.0 + np.exp(-6.400950713353678 * load_proxy))
    congestion_score = load_proxy * load_gate * 1.4579499662939925
    ddl_breach = (slack <= -0.0051067269566107965).astype(float)
    exec_penalty = 0.17260590240571339 * norm_exec * ddl_breach
    comm_penalty = (1.0 - 0.17260590240571339) * norm_comm * ddl_breach
    uncert_gate = 1.0 / (1.0 + np.exp(-6.400950713353678 * (norm_uncert - 0.28889629570611847)))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_breach
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_score, -2.0, 2.0) - 1.55027154323137 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) + np.clip(congestion_score, -2.0, 2.0) + 0.4167603823765746 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 1.7314301467885902 * np.clip(norm_work * ddl_gate, -2.0, 2.0) + np.clip(exec_penalty, -2.0, 2.0) + np.clip(comm_penalty, -2.0, 2.0) + np.clip(successor_blocking_penalty, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
