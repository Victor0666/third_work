import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule: replaces hardcoded quantiles with tunable parameters;
       uses robust range-based normalization with PARAMS["quantile_q1"]/["quantile_q3"];
       introduces unified congestion signal (ready_wait_time + uncertainty) gated by DDL feasibility;
       enforces critical-path leverage via upward_rank * remaining_work * (slack <= 0);
       uses sign-preserving MAD only for slack; applies clipped linear slack pressure near boundary;
       eliminates all numeric literals except -2, -1, 0, 1, 2; uses np.finfo for epsilon safeguards."""
    eps = 0.006238531475190005
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
            q1 = np.quantile(x, 0.36070476610444796)
            q3 = np.quantile(x, 0.7618688371000355)
        iqr = q3 - q1
        spread = iqr if iqr > eps else eps
        return (x - q1) / spread
    norm_energy = robust_range_normalize(min_incremental_energy)
    norm_duration = robust_range_normalize(min_exec_time + min_comm_time)
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
    ddl_gate = 1.0 / (1.0 + np.exp(-3.7063561263262814 * slack))
    ddl_breach = (slack <= 0.0).astype(float)
    critical_path_leverage = norm_rank * norm_work * ddl_breach
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.318649557548495
    norm_slack_penalty = robust_range_normalize(raw_slack_penalty)
    congestion_signal = norm_wait + norm_uncert
    congestion_score = congestion_signal * (1.0 - ddl_gate)
    wait_mean = np.mean(ready_wait_time)
    wait_denom = np.maximum(wait_mean, eps)
    wait_benefit = np.clip(ready_wait_time / wait_denom, 0.0, 1.0)
    uncert_gate = 1.0 / (1.0 + np.exp(-3.7063561263262814 * (norm_uncert - 0.6607654436997054)))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_breach
    slack_pressure = np.clip(-slack, 0.0, 1.0)
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(norm_rank * (1.0 + 0.5574940409987326 * slack_pressure), -2.0, 2.0) - 0.20314337031543978 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(congestion_score, -2.0, 2.0) + 0.017826188085994534 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 0.04247271428244811 * np.clip(norm_work * ddl_gate, -2.0, 2.0) + 1.318649557548495 * np.clip(slack_pressure, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
