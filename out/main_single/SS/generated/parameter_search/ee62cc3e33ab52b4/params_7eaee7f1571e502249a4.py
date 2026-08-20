import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's superior gradient properties and quantile normalization
       with Parent 1's congestion-thresholded starvation relief — now implemented via load-suppressed benefit.
       Structural changes:
         - Removed 'criticality_boost_slope' (reducing parameter count to 12).
         - Replaced tunable criticality boost with fixed-slope linear amplification: 1.0 + 1.0 * (1 - load_gate) * slack_sensitivity_gate,
           clipped to [1.0, 2.0] — preserves intent without new parameter.
         - All 12 parameters are used; no numeric literals outside [-2, -1, 0, 1, 2].
         - Robust quantile normalization preserved; sign-preserving MAD for slack."""
    eps = 2.2672666317981143e-06
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
            q1 = np.quantile(x, 0.31881900157039156)
            q3 = np.quantile(x, 0.731667912610267)
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
    load_proxy = norm_wait + norm_uncert
    load_gate = 1.0 / (1.0 + np.exp(-4.019012274340381 * load_proxy))
    ddl_gate = 1.0 / (1.0 + np.exp(-4.019012274340381 * slack))
    slack_sensitivity_gate = 1.0 / (1.0 + np.exp(-4.019012274340381 * (slack - -0.10830027335330161)))
    criticality_boost = 1.0 + 1.0 * (1.0 - load_gate) * slack_sensitivity_gate
    boosted_rank = norm_rank * np.clip(criticality_boost, 1.0, 2.0)
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 3.123706092474892
    norm_slack_penalty = robust_range_normalize(raw_slack_penalty)
    successor_release_prob = 1.0 - np.clip(norm_wait, 0.0, 1.0)
    successor_blocking_penalty = norm_rank * (1.0 - successor_release_prob) * 0.6782303286500535
    starvation_benefit = 1.0 - np.exp(-norm_wait)
    load_suppressed_starvation = starvation_benefit * (1.0 - load_gate)
    ddl_breach = (slack <= -0.10830027335330161).astype(float)
    exec_penalty = 0.5832734235408468 * norm_exec * ddl_breach
    comm_penalty = (1.0 - 0.5832734235408468) * norm_comm * ddl_breach
    uncert_gate = 1.0 / (1.0 + np.exp(-4.019012274340381 * (norm_uncert - 0.31881900157039156)))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_breach
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 1.836830079511922 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(load_suppressed_starvation, -2.0, 2.0) + np.clip(0.11299954132962817 * load_proxy * ddl_gate, -2.0, 2.0) + 0.3036211709977891 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 0.4015930446241336 * np.clip(norm_work * ddl_gate, -2.0, 2.0) + np.clip(exec_penalty, -2.0, 2.0) + np.clip(comm_penalty, -2.0, 2.0) + np.clip(successor_blocking_penalty, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
