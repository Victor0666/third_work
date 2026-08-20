import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule: replaces median-MAD with quantile-based robust normalization (Q1/Q3);
       introduces joint congestion signal (ready_wait_time + uncertainty) gated by both DDL feasibility and high-congestion threshold;
       removes duration_robustness and wait_saturation_offset (evidence shows low activity/inactivity);
       uses sign-preserving MAD only for slack to sharpen urgency near violation;
       adds conditional ddl_protection_gate activated only for slack >= 0 to enable energy/uncertainty penalties exclusively under feasibility;
       enforces critical-path awareness *only* under deadline pressure via upward_rank × remaining_work × ddl_gate;
       eliminates redundant critical_path_booster and linear rank-slack coupling; retains bounded AST depth and <=8 feature interactions."""
    eps = 5.7125900700304216e-05
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
            q1 = np.quantile(x, 0.21108248408340108)
            q3 = np.quantile(x, 0.8338240717977599)
        iqr = q3 - q1 if q3 - q1 > eps else eps
        return (x - q1) / iqr

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
    norm_energy = quantile_normalize(min_incremental_energy)
    norm_duration = quantile_normalize(min_exec_time + min_comm_time)
    norm_rank = quantile_normalize(upward_rank)
    norm_work = quantile_normalize(remaining_work)
    norm_wait = quantile_normalize(ready_wait_time)
    norm_uncert = quantile_normalize(uncertainty)
    norm_slack = mad_normalize_slack(slack)
    ddl_feasible = (slack >= 0.0).astype(float)
    ddl_protection_gate = 1.0 / (1.0 + np.exp(-6.6661613403536295 * (slack - 0.0)))
    ddl_pressure = (slack <= 0.0).astype(float)
    critical_leverage = norm_rank * norm_work * ddl_pressure
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.06703228415033
    norm_slack_penalty = quantile_normalize(raw_slack_penalty)
    congestion = ready_wait_time + uncertainty
    congestion_gate = (congestion > 0.0026478872239494218).astype(float) * ddl_feasible
    norm_congestion = quantile_normalize(congestion) * congestion_gate
    wait_benefit = np.clip(0.3459803073789355 * ready_wait_time, 0.0, 2.0)
    uncert_gate = (uncertainty > 0.0026478872239494218).astype(float) * ddl_feasible
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_leverage, -2.0, 2.0) - np.clip(norm_rank * (1.0 + 1.546573008167424 * np.clip(-norm_slack, 0.0, 1.0)), -2.0, 2.0) - 1.330083704564999 * np.clip(norm_energy * ddl_protection_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(norm_congestion, -2.0, 2.0) + 0.0 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 0.4697432026142482 * np.clip(norm_work * ddl_pressure, -2.0, 2.0) + np.clip(norm_slack * 1.6974961539896642, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
