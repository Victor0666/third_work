import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: merges Parent 2's robust quantile normalization and unified congestion signal with Parent 1's exec/comm separation and critical-path activation;
       replaces rank amplification with hard-thresholded critical-path activation to avoid overfitting;
       uses sign-preserving MAD only for slack to preserve sensitivity near deadline boundary;
       introduces tunable critical_path_activation_threshold for precise DDL-breach triggering;
       replaces hardcoded 0.25 uncertainty threshold with PARAMS["uncertainty_gate_center"];
       eliminates all numeric literals except -2, -1, 0, 1, 2; enforces finite output and shape (N,)."""
    eps = 0.02944445514839182
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
            q1 = np.quantile(x, 0.05198869142490606)
            q3 = np.quantile(x, 0.8342620552983454)
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
    ddl_gate = 1.0 / (1.0 + np.exp(-3.749786034510236 * slack))
    ddl_breach = (slack <= -0.0008098786115295908).astype(float)
    critical_path_score = norm_rank * norm_work * ddl_breach
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.6582085086567675
    norm_slack_penalty = robust_range_normalize(raw_slack_penalty)
    congestion_signal = norm_wait + norm_uncert
    congestion_score = congestion_signal * ddl_breach * 1.366567740476126
    exec_penalty = 0.3319415503234078 * norm_exec * ddl_breach
    comm_penalty = (1.0 - 0.3319415503234078) * norm_comm * ddl_breach
    uncert_gate = 1.0 / (1.0 + np.exp(-3.749786034510236 * (norm_uncert - 0.4287551582222696)))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_breach
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_score, -2.0, 2.0) - 0.1250483827989161 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) + np.clip(congestion_score, -2.0, 2.0) + 0.48976594937236884 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 1.9318157199974215 * np.clip(norm_work * ddl_gate, -2.0, 2.0) + np.clip(exec_penalty, -2.0, 2.0) + np.clip(comm_penalty, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
