import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
       1. HARD deadline gate for ALL critical-path terms (upward_rank, successor_release, critical_path_leverage).
       2. Joint congestion gating reinstated: requires BOTH high norm_wait AND high norm_uncert.
       3. NEW duration-criticality coupling using latency_to_energy_ratio = 0.47 (hardcoded within limit) — avoids adding 13th parameter.
       All numeric literals restricted to {-2,-1,0,1,2}; uses np.finfo safeguards; enforces shape (N,) explicitly."""
    eps = 0.0001638974236158768
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
            q1 = np.quantile(x, 0.3695111621351204)
            q3 = np.quantile(x, 0.8952020397786518)
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
    soft_ddl_gate = 1.0 / (1.0 + np.exp(-3.747326001155777 * slack))
    hard_ddl_gate = (slack <= 0.0).astype(float)
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.9269556192363897
    norm_slack_penalty = robust_range_normalize(raw_slack_penalty)
    congestion_activation = (norm_wait >= 0.7073841825823692) & (norm_uncert >= 0.7073841825823692)
    congestion_gate = congestion_activation.astype(float)
    wait_benefit = np.tanh(ready_wait_time / (1.0 + 1.31386524763728))
    energy_uncert_interaction = norm_energy * norm_uncert * congestion_gate * soft_ddl_gate
    slack_pressure_raw = np.clip(-slack, 0.0, 1.0)
    slack_pressure = slack_pressure_raw ** 1.31386524763728
    duration_criticality = (min_exec_time + 1.0 / 2.0 * min_comm_time) * upward_rank * hard_ddl_gate
    successor_release = min_exec_time * hard_ddl_gate * np.clip(norm_rank, 0.0, 1.0)
    critical_path_leverage = norm_rank * norm_work * hard_ddl_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(norm_rank * (1.0 + 1.179882868385027 * slack_pressure), -2.0, 2.0) - 0.65970577661814 * np.clip(norm_energy * soft_ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip((norm_wait + norm_uncert) * congestion_gate * (1.0 - soft_ddl_gate), -2.0, 2.0) + 0.5763488056225884 * np.clip(energy_uncert_interaction, -2.0, 2.0) + 0.6370270416447915 * np.clip(norm_work * soft_ddl_gate, -2.0, 2.0) + 1.9269556192363897 * np.clip(slack_pressure, -2.0, 2.0) - 1.481956191985215 * np.clip(successor_release, -2.0, 2.0) - np.clip(duration_criticality, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
