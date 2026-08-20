import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule: replaces median-MAD with robust sign-preserving range normalization;
       introduces unified congestion signal (ready_wait_time + uncertainty) gated by DDL feasibility;
       removes redundant duration_robustness and wait_saturation_offset per counterfactual evidence;
       enforces strict sign-consistent prioritization: smaller score = higher priority;
       uses clipped linear slack pressure near violation boundary for numerical stability;
       incorporates upward_rank * remaining_work interaction only under DDL breach per diagnostic evidence."""
    eps = 0.008495870920240347
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
            rng = eps
            mid = x[0]
        else:
            p_low = np.percentile(x, 13.79058255916479)
            p_high = np.percentile(x, 90.44298681703965)
            rng = p_high - p_low
            mid = (p_low + p_high) / 2.0
        spread = rng if rng > eps else eps
        return (x - mid) / spread
    norm_slack = robust_range_normalize(slack)
    norm_energy = robust_range_normalize(min_incremental_energy)
    norm_rank = robust_range_normalize(upward_rank)
    norm_work = robust_range_normalize(remaining_work)
    norm_wait = robust_range_normalize(ready_wait_time)
    norm_uncert = robust_range_normalize(uncertainty)
    ddl_gate = 1.0 / (1.0 + np.exp(-8.969840027552756 * slack))
    ddl_breach = (slack <= 0.0).astype(float)
    critical_path_leverage = norm_rank * norm_work * ddl_breach
    slack_pressure = np.clip(-slack, 0.0, 1.0)
    congestion = (ready_wait_time + uncertainty) * ddl_gate
    wait_benefit = 1.0 - np.exp(-0.7700213705444905 * ready_wait_time)
    uncert_gate = 1.0 / (1.0 + np.exp(-8.969840027552756 * (uncertainty - 0.10502301733787936)))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    energy_slack_penalty = norm_energy * (1.0 + 2.696805237874053 * slack_pressure) * ddl_gate
    score = +np.clip(slack_pressure, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(norm_rank * (1.0 + 1.04362455387981 * slack_pressure), -2.0, 2.0) - 0.20786751382727792 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(congestion, -2.0, 2.0) + 0.24951910462932364 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 2.696805237874053 * np.clip(energy_slack_penalty, -2.0, 2.0) + 1.2538444284720245 * np.clip(norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
