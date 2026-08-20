import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule: uses all declared parameters; replaces sigmoid gates with robust clipped-MAD slack gating;
       integrates successor-release interaction via min_exec_time × upward_rank × (slack <= 0);
       applies strict feasibility-gating (slack >= 0) to all energy/uncertainty terms;
       leverages validated 'add_conditional_ddl_protection_gate' and 'add_upward_rank_remaining_work_interaction';
       uses MAD-normalized slack [-2,2] as primary urgency signal;
       avoids unbounded ops, ensures finite output and shape (N,)."""
    eps = 2.72521323729762e-05
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
        spread = mad if mad > eps else eps
        return (x - med) / spread
    norm_slack = median_mad_normalize(slack)
    norm_energy = median_mad_normalize(min_incremental_energy)
    norm_exec = median_mad_normalize(min_exec_time)
    norm_comm = median_mad_normalize(min_comm_time)
    norm_rank = median_mad_normalize(upward_rank)
    norm_work = median_mad_normalize(remaining_work)
    norm_wait = median_mad_normalize(ready_wait_time)
    norm_uncert = median_mad_normalize(uncertainty)
    urgency = np.clip(norm_slack, -2.0, 2.0)
    ddl_gate = 1.0 / (1.0 + np.exp(-6.739860621511693 * slack))
    ddl_breach = (slack <= 0.0).astype(float)
    critical_leverage = norm_rank * norm_work * ddl_breach
    successor_release = norm_exec * norm_rank * ddl_breach
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.243923509490825
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-3.762496933809763 * (slack_pressure - 1.0)))
    rank_boost = norm_rank * (1.0 + 1.3911411326942065 * rank_gate) * ddl_breach
    energy_penalty = 0.6131097187155994 * norm_energy * ddl_gate
    uncert_gate = 1.0 / (1.0 + np.exp(-6.739860621511693 * (norm_uncert - 0.6483466359307954)))
    energy_uncert_penalty = 0.3392142565910725 * norm_energy * norm_uncert * ddl_gate * uncert_gate
    wait_relief = 1.0 - np.exp(-0.3323818317579463 * norm_wait)
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_leverage, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) + np.clip(rank_boost, -2.0, 2.0) + energy_penalty + energy_uncert_penalty - np.clip(wait_relief, -2.0, 2.0) + 0.8988762750523613 * np.clip(norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
