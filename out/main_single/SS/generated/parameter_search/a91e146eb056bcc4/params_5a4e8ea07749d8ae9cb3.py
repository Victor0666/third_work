import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule: removes unused parameters; retains successor-release interaction;
       uses joint congestion gating (high ready_wait_time AND high uncertainty);
       applies energy/uncertainty penalties only under feasibility (slack >= 0);
       replaces sigmoid gates with direct threshold-based logic for stability and evidence alignment."""
    eps = 4.3199535300043545e-06
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def robust_mad_normalize(x):
        x = np.copy(x)
        if N == 1:
            med = x[0]
            mad = eps
        else:
            med = np.median(x)
            mad = np.median(np.abs(x - med))
        spread = np.maximum(mad, eps)
        return (x - med) / spread
    norm_slack = robust_mad_normalize(slack)
    norm_energy = robust_mad_normalize(min_incremental_energy)
    norm_rank = robust_mad_normalize(upward_rank)
    norm_work = robust_mad_normalize(remaining_work)
    norm_wait = robust_mad_normalize(ready_wait_time)
    norm_uncert = robust_mad_normalize(uncertainty)
    feasible_mask = (slack >= 0.0).astype(float)
    ddl_breach = (slack <= 0.0).astype(float)
    critical_path_leverage = norm_rank * norm_work * ddl_breach
    successor_release = np.clip(min_exec_time, eps, None) * np.clip(norm_rank, -1.0, 1.0) * ddl_breach
    slack_mad = np.maximum(np.median(np.abs(slack - np.median(slack))), eps)
    norm_slack_urgency = (slack - np.median(slack)) / (slack_mad + eps)
    raw_slack_penalty = np.where(slack < 0.0, (-slack) ** 1.5336306041656096, 0.0)
    norm_slack_penalty = robust_mad_normalize(raw_slack_penalty)
    wait_centered = ready_wait_time - np.median(ready_wait_time)
    wait_std = np.maximum(np.std(ready_wait_time), eps)
    wait_z = wait_centered / (wait_std + eps)
    uncert_centered = uncertainty - np.median(uncertainty)
    uncert_std = np.maximum(np.std(uncertainty), eps)
    uncert_z = uncert_centered / (uncert_std + eps)
    congestion_gate = (wait_z > 0.04843811402386472) & (uncert_z > 0.04843811402386472)
    congestion_gate = congestion_gate.astype(float)
    wait_benefit = np.exp(-0.24590644201113943 * ready_wait_time) * (1.0 - feasible_mask)
    energy_uncert_penalty = norm_energy * norm_uncert * feasible_mask * congestion_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - np.clip(norm_slack_urgency, -2.0, 2.0) - 0.31417699854660897 * np.clip(norm_energy * feasible_mask, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.3716140747675907 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 0.12681692850988657 * np.clip(norm_work, -2.0, 2.0) + 0.7167526801530996 * np.clip(norm_rank * feasible_mask, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
