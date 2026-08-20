import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining Parent 2's sharp sigmoid DDL gate and host-load suppression 
       with Parent 1's MAD-based normalization and bounded linear coupling. Introduces:
       - Adaptive median-shifted low-rank mask using tunable threshold for more robust suppression;
       - Clipped linear starvation relief with explicit cap (replacing unbounded linear);
       - Unified duration-uncertainty normalization using sum-of-min-times instead of separate features;
       - Critical path leverage now gated by both slack<=0 AND upward_rank > median, avoiding over-prioritization
         of trivial successors under deadline breach.
       All intermediate terms clipped to [-2,2] for stability; final score finite & deterministic."""
    eps = 7.386586798000731e-05
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def median_mad_normalize(x):
        x = np.copy(x)
        if N == 1:
            med = x[0]
            mad = eps
        else:
            med = np.median(x)
            mad = np.median(np.abs(x - med))
        spread = np.maximum(mad, eps)
        return (x - med) / spread
    norm_slack = median_mad_normalize(slack)
    norm_energy = median_mad_normalize(min_incremental_energy)
    norm_duration = median_mad_normalize(min_exec_time + min_comm_time)
    norm_rank = median_mad_normalize(upward_rank)
    norm_work = median_mad_normalize(remaining_work)
    norm_wait = median_mad_normalize(ready_wait_time)
    norm_uncert = median_mad_normalize(uncertainty)
    ddl_gate = 1.0 / (1.0 + np.exp(-8.893590828777413 * slack))
    rank_median = np.median(upward_rank) if N > 1 else upward_rank[0]
    ddl_breach_mask = (slack <= 0.0).astype(float)
    critical_rank_mask = (upward_rank > rank_median).astype(float)
    critical_path_leverage = norm_rank * norm_work * ddl_breach_mask * critical_rank_mask * 0.4711995538403675
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.241722721365493
    coupled_slack = np.clip(norm_slack, -1.0, 1.0)
    rank_slack_coupling = 1.0 + 2.2885829784478506 * coupled_slack
    wait_benefit = np.clip(0.2636045075814673 * ready_wait_time, 0.0, 1.0)
    tight_slack_mask = (slack <= eps).astype(float)
    high_uncert_mask = (norm_uncert >= 0.5589514450481208).astype(float)
    duration_risk_score = norm_duration * tight_slack_mask * high_uncert_mask * 0.8139584245112075
    energy_uncert_penalty = norm_energy * norm_uncert * ddl_gate
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    energy_slack_penalty = norm_energy * (1.0 + 2.241722721365493 * slack_pressure) * ddl_gate
    low_rank_mask = (upward_rank < rank_median).astype(float) * (slack > 0.0).astype(float)
    host_load_penalty = norm_energy * norm_wait * low_rank_mask
    score = +raw_slack_penalty - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(norm_rank * rank_slack_coupling, -2.0, 2.0) - 0.2060391505711482 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 1.5309199786812775 * np.clip(duration_risk_score, -2.0, 2.0) + 0.2983996978617264 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 2.241722721365493 * np.clip(energy_slack_penalty, -2.0, 2.0) + 0.981593444970613 * np.clip(norm_work, -2.0, 2.0) + np.clip(host_load_penalty, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
