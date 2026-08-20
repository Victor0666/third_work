import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule: replaces hidden 0.5 with PARAMS["rank_boost_lower_bound"];
       uses robust range-based normalization;
       introduces unified congestion signal (ready_wait_time + uncertainty) gated by DDL feasibility;
       enforces critical path leverage only under strict slack <= 0;
       replaces multiplicative ddl_gate with additive binary gate for numerical stability;
       all terms clipped to [-2,2] and fused with bounded coefficients."""
    eps = 0.0005489724504165144
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
            xmin = xmax = x[0]
        else:
            xmin = np.min(x)
            xmax = np.max(x)
        spread = xmax - xmin if xmax - xmin > eps else eps
        return (x - xmin) / spread
    norm_slack = robust_range_normalize(slack)
    norm_energy = robust_range_normalize(min_incremental_energy)
    norm_rank = robust_range_normalize(upward_rank)
    norm_work = robust_range_normalize(remaining_work)
    norm_wait = robust_range_normalize(ready_wait_time)
    norm_uncert = robust_range_normalize(uncertainty)
    congestion = ready_wait_time + uncertainty
    norm_congestion = robust_range_normalize(congestion)
    ddl_gate = (slack <= 0.0).astype(float)
    critical_path_leverage = norm_rank * norm_work * ddl_gate
    slack_pressure = np.clip(1.0 - norm_slack, 0.0, 1.0)
    wait_benefit = 1.0 - np.exp(-0.48113200918970694 * ready_wait_time)
    congestion_gate = 1.0 / (1.0 + np.exp(-3.82651485275116 * (norm_congestion - 0.45720388337418205)))
    energy_uncert_penalty = norm_energy * norm_uncert * congestion_gate * ddl_gate
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.272681200392677
    norm_slack_penalty = robust_range_normalize(raw_slack_penalty)
    rank_boost = 1.0 + 0.7356617435282473 * slack_pressure
    rank_boost = np.clip(rank_boost, 0.4164389062494045, 2.6782073926483205)
    boosted_rank = norm_rank * rank_boost
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) - 0.18332252340600536 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) + 0.8466446197697072 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 0.7468603002929395 * np.clip(norm_work, -2.0, 2.0) + np.clip(norm_congestion * ddl_gate, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
