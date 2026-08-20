import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule: combines Parent 2's hard DDL gate & congestion-aware risk gating with Parent 1's unified duration-uncertainty coupling;
       introduces *dual-gated critical path leverage*: activated only under both (slack <= 0) AND (congestion > threshold);
       replaces raw_slack_penalty with *adaptive power-law* using normalized slack offset to avoid zero-power degeneracy;
       adds duration-uncertainty interaction as distinct term (not just in energy_uncert_penalty) to penalize long uncertain delays;
       retains all robust clipping, range normalization, and shape safeguards."""
    eps = 3.668324851549434e-06
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
    risk_gate = 1.0 / (1.0 + np.exp(-2.052660842922262 * (norm_congestion - 0.4680632667966418)))
    dual_gate = ddl_gate * risk_gate
    critical_path_leverage = norm_rank * norm_work * dual_gate
    adaptive_slack_base = np.clip(1.0 - norm_slack + eps, eps, 2.0)
    slack_pressure = adaptive_slack_base ** 1.076067425559333
    norm_slack_pressure = robust_range_normalize(slack_pressure)
    wait_benefit = 1.0 - np.exp(-1.089311452427484e-05 * ready_wait_time)
    energy_uncert_penalty = norm_energy * norm_uncert * dual_gate
    duration = min_exec_time + min_comm_time
    norm_duration = robust_range_normalize(duration)
    duration_uncert_penalty = norm_duration * norm_uncert * dual_gate
    rank_boost = 1.0 + 1.9266899281174696 * norm_slack_pressure
    rank_boost = np.clip(rank_boost, 0.5077994415296081, 1.0054554514183496)
    boosted_rank = norm_rank * rank_boost
    score = +np.clip(norm_slack_pressure, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) - 1.0678302211350084 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) + 0.8765926150045309 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 0.532144585560484 * np.clip(duration_uncert_penalty, -2.0, 2.0) + 0.6359242061147328 * np.clip(norm_work * ddl_gate, -2.0, 2.0) + np.clip(norm_congestion * ddl_gate, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
