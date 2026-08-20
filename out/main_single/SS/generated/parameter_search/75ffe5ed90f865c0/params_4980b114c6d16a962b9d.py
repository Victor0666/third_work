import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule: replaces median-MAD with quantile-based robust normalization (Q1/Q3);
       introduces joint congestion signal (ready_wait_time + uncertainty) gated by both DDL feasibility and high-congestion threshold;
       removes redundant duration_robustness and wait_saturation_offset; replaces coupled_slack with sign-preserving MAD-normalized slack;
       adds conditional ddl_protection_gate activated only for slack >= 0 to isolate energy/uncertainty penalties to feasible region;
       enforces critical-path awareness via upward_rank × remaining_work × ddl_gate only under deadline pressure."""
    eps = 7.317880570316888e-05
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
            iqr = eps
        else:
            q1 = np.quantile(x, 0.21391051479477208)
            q3 = np.quantile(x, 0.6546890839011072)
            iqr = q3 - q1
        spread = iqr if iqr > eps else eps
        return (x - q1) / spread
    norm_slack = quantile_normalize(slack)
    norm_energy = quantile_normalize(min_incremental_energy)
    norm_duration = quantile_normalize(min_exec_time + min_comm_time)
    norm_rank = quantile_normalize(upward_rank)
    norm_work = quantile_normalize(remaining_work)
    norm_wait = quantile_normalize(ready_wait_time)
    norm_uncert = quantile_normalize(uncertainty)
    slack_mad = np.median(np.abs(slack - np.median(slack)))
    slack_mad = max(slack_mad, eps)
    norm_slack_sharp = (slack - np.median(slack)) / slack_mad
    ddl_feasible = (slack >= 0.0).astype(float)
    ddl_protection_gate = 1.0 / (1.0 + np.exp(-8.001747992292124 * (slack + eps)))
    critical_path_leverage = norm_rank * norm_work * ddl_feasible
    raw_neg_slack = np.maximum(-slack, 0.0)
    slack_penalty = raw_neg_slack ** 1.5375923646408691
    norm_slack_penalty = quantile_normalize(slack_penalty)
    congestion = ready_wait_time + uncertainty
    congestion_norm = quantile_normalize(congestion)
    congestion_gate = (congestion_norm > 0.01265298239698194).astype(float)
    congestion_penalty = congestion_norm * congestion_gate * ddl_feasible
    wait_benefit = np.clip(0.29262932858761426 * ready_wait_time, 0.0, 2.0)
    energy_uncert_interaction = norm_energy * norm_uncert * congestion_gate * ddl_feasible
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(norm_rank, -2.0, 2.0) * (1.0 + 0.8821294354887083 * np.clip(-norm_slack_sharp, 0.0, 1.0)) - 0.4073999931668415 * np.clip(norm_energy * ddl_protection_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(congestion_penalty, -2.0, 2.0) + 0.2665118586420658 * np.clip(energy_uncert_interaction, -2.0, 2.0) + 1.1230664835993185 * np.clip(norm_work, -2.0, 2.0) + np.clip(2.1910630299713656 * np.clip(-norm_slack_sharp, 0.0, 2.0), -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
