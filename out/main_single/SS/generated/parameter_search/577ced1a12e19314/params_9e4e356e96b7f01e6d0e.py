import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule: replaces median-MAD with quantile-based robust normalization;
       introduces congestion-aware gate using `ready_wait_time + uncertainty`;
       removes redundant `duration_robustness` and `wait_saturation_offset` per evidence;
       enforces sign-preserving MAD only on slack; uses clipped linear `wait_benefit`;
       adds conditional `ddl_protection_gate` activated only for slack >= 0;
       combines critical path leverage via `upward_rank * remaining_work` gated by DDL feasibility."""
    eps = 2.2373707906636335e-06
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
            q1 = x[0]
            q3 = x[0]
        else:
            q1 = np.quantile(x, 0.2708264562999976)
            q3 = np.quantile(x, 0.804468143716949)
        iqr = q3 - q1 if q3 - q1 > eps else eps
        return (x - (q1 + q3) / 2.0) / iqr

    def mad_normalize_slack(x):
        x = np.copy(x)
        if N == 1:
            med = x[0]
            mad = eps
        else:
            med = np.median(x)
            mad = np.median(np.abs(x - med))
        spread = mad if mad > eps else eps
        return (x - med) / spread
    norm_slack = mad_normalize_slack(slack)
    norm_energy = quantile_normalize(min_incremental_energy)
    norm_duration = quantile_normalize(min_exec_time + min_comm_time)
    norm_rank = quantile_normalize(upward_rank)
    norm_work = quantile_normalize(remaining_work)
    norm_wait = quantile_normalize(ready_wait_time)
    norm_uncert = quantile_normalize(uncertainty)
    ddl_feasible = (slack >= 0.0).astype(float)
    ddl_protection_gate = ddl_feasible * (1.0 / (1.0 + np.exp(-3.2007517019933767 * (slack - 0.0))))
    critical_leverage = norm_rank * norm_work * ddl_feasible
    raw_slack_risk = np.maximum(-slack, 0.0) ** 2.9113163369391737
    norm_slack_risk = quantile_normalize(raw_slack_risk)
    congestion = ready_wait_time + uncertainty
    congestion_gate = ddl_feasible * (1.0 / (1.0 + np.exp(-5.173534659669258 * (congestion - 0.02046527357160177 * np.quantile(congestion, 0.804468143716949)))))
    wait_benefit = np.clip(0.7885198070239774 * ready_wait_time, 0.0, 1.0)
    energy_uncert_interaction = norm_energy * norm_uncert * ddl_protection_gate * congestion_gate
    score = +norm_slack_risk - critical_leverage - norm_rank * (1.0 + 1.1648923093264467 * np.clip(-norm_slack, 0.0, 1.0)) - 0.1889329982990645 * norm_energy * ddl_protection_gate - wait_benefit + congestion_gate * norm_duration + 0.4686676444469846 * energy_uncert_interaction + 0.06833447774616576 * norm_work * ddl_feasible
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
