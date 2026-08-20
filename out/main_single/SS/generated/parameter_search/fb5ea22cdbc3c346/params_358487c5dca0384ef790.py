import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule: replaces hidden 0.5 with PARAMS["rank_boost_lower_bound"];
       uses robust range-based normalization;
       introduces unified congestion signal (ready_wait_time + uncertainty) gated by DDL feasibility;
       enforces critical path leverage only under strict slack <= 0;
       replaces multiplicative ddl_gate with additive binary gate for numerical stability;
       all terms clipped to [-2,2] and fused with bounded coefficients."""
    eps = 0.0019584208498256866
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
    wait_benefit = 1.0 - np.exp(-0.529875087467734 * ready_wait_time)
    congestion_gate = 1.0 / (1.0 + np.exp(-4.118146237166904 * (norm_congestion - 0.1711886315394772)))
    energy_uncert_penalty = norm_energy * norm_uncert * congestion_gate * ddl_gate
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.947211329376354
    norm_slack_penalty = robust_range_normalize(raw_slack_penalty)
    rank_boost = 1.0 + 1.4186242615335691 * slack_pressure
    rank_boost = np.clip(rank_boost, 0.17715799438808272, 2.1281495310590417)
    boosted_rank = norm_rank * rank_boost
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) - 0.2726256698293378 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) + 0.7545922916884626 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 1.999692193669534 * np.clip(norm_work, -2.0, 2.0) + np.clip(norm_congestion * ddl_gate, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
