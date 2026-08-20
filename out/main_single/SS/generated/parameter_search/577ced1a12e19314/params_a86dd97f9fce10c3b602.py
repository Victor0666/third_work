import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule: replaces median-MAD with quantile-based robust normalization;
       introduces congestion-aware gate using `ready_wait_time + uncertainty`;
       removes redundant `duration_robustness` and `wait_saturation_offset` per evidence;
       enforces sign-preserving MAD only on slack; uses clipped linear `wait_benefit`;
       adds conditional `ddl_protection_gate` activated only for slack >= 0;
       combines critical path leverage via `upward_rank * remaining_work` gated by DDL feasibility."""
    eps = 4.933888115788104e-06
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
            q1 = np.quantile(x, 0.11987926929748251)
            q3 = np.quantile(x, 0.6913664657098902)
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
    ddl_protection_gate = ddl_feasible * (1.0 / (1.0 + np.exp(-4.822129372130846 * (slack - 0.0))))
    critical_leverage = norm_rank * norm_work * ddl_feasible
    raw_slack_risk = np.maximum(-slack, 0.0) ** 3.6175076722702197
    norm_slack_risk = quantile_normalize(raw_slack_risk)
    congestion = ready_wait_time + uncertainty
    congestion_gate = ddl_feasible * (1.0 / (1.0 + np.exp(-1.527649707551241 * (congestion - 0.6799428919407255 * np.quantile(congestion, 0.6913664657098902)))))
    wait_benefit = np.clip(0.0640986921887971 * ready_wait_time, 0.0, 1.0)
    energy_uncert_interaction = norm_energy * norm_uncert * ddl_protection_gate * congestion_gate
    score = +norm_slack_risk - critical_leverage - norm_rank * (1.0 + 0.9733505310061427 * np.clip(-norm_slack, 0.0, 1.0)) - 0.21963906004048978 * norm_energy * ddl_protection_gate - wait_benefit + congestion_gate * norm_duration + 0.4807220922825167 * energy_uncert_interaction + 0.23493502407231603 * norm_work * ddl_feasible
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
