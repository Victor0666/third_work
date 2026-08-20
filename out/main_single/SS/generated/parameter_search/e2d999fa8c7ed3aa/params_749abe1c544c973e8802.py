import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule: replaces hardcoded 0.25/0.75 with tunable quantiles;
       uses sign-preserving MAD-normalized slack for sharp deadline violation sensitivity;
       applies DDL-protection gate only for slack >= 0 to enable energy/uncertainty penalties exclusively when feasible;
       includes upward_rank × remaining_work interaction under deadline pressure;
       introduces congestion signal (ready_wait_time + uncertainty) gated by DDL feasibility and uncertainty threshold;
       uses clipped linear wait_benefit for low-wait discriminability."""
    eps = 1.0144498339939582e-05
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def quantile_normalize(x):
        x = np.copy(x)
        if N == 1:
            q1 = q3 = x[0]
        else:
            q1 = np.quantile(x, 0.316720532745398)
            q3 = np.quantile(x, 0.8799635423915859)
        iqr = q3 - q1 if q3 > q1 else eps
        return (x - q1) / iqr
    norm_slack = quantile_normalize(slack)
    norm_energy = quantile_normalize(min_incremental_energy)
    norm_duration = quantile_normalize(min_exec_time + min_comm_time)
    norm_rank = quantile_normalize(upward_rank)
    norm_work = quantile_normalize(remaining_work)
    norm_wait = quantile_normalize(ready_wait_time)
    norm_uncert = quantile_normalize(uncertainty)
    slack_med = np.median(slack)
    slack_mad = np.median(np.abs(slack - slack_med))
    slack_spread = slack_mad if slack_mad > eps else eps
    norm_slack_mad = (slack - slack_med) / slack_spread
    ddl_feasible = (slack >= 0.0).astype(float)
    ddl_protection_gate = 1.0 / (1.0 + np.exp(-2.116608740419646 * (slack - 0.0)))
    ddl_pressure = (slack <= 0.0).astype(float)
    critical_leverage = norm_rank * norm_work * ddl_pressure
    congestion = ready_wait_time + uncertainty
    congestion_norm = quantile_normalize(congestion)
    congestion_gate = 1.0 / (1.0 + np.exp(-2.116608740419646 * (congestion_norm - 0.1869207942688525)))
    congestion_signal = congestion_norm * congestion_gate * ddl_feasible
    wait_benefit = np.clip(0.8647019529883584 * ready_wait_time, 0.0, 2.0)
    energy_uncert_coupling = norm_energy * norm_uncert * congestion_gate * ddl_feasible
    slack_pressure = np.clip(-norm_slack_mad, 0.0, 2.0)
    energy_slack_amplifier = 1.0 + 3.9866443696296465 * slack_pressure
    score = +np.clip(norm_slack_mad, -2.0, 2.0) - np.clip(critical_leverage, -2.0, 2.0) - np.clip(norm_rank, -2.0, 2.0) - 0.7235732631551889 * np.clip(norm_energy * ddl_protection_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(congestion_signal, -2.0, 2.0) + 0.5994682387275047 * np.clip(energy_uncert_coupling, -2.0, 2.0) + np.clip(norm_energy * energy_slack_amplifier * ddl_feasible, -2.0, 2.0) + 0.11369257094822455 * np.clip(norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
