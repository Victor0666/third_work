import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: replaces sigmoid ddl_gate with piecewise linear feasibility gate for interpretable hard deadline enforcement;
       introduces critical-path release interaction (upward_rank × remaining_work × (1 − ddl_gate)) to prioritize unblocking successors only when DDL-feasible;
       adds load-aware anti-starvation gate activated only under congestion and prolonged wait — eliminating redundant parameters while improving fairness;
       retains median-MAD normalization for robustness; clips all terms to [-2,2]; enforces deterministic finite output."""
    eps = 2.3850188843971277e-05
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
    total_duration = min_exec_time + min_comm_time
    norm_duration = median_mad_normalize(total_duration)
    norm_rank = median_mad_normalize(upward_rank)
    norm_work = median_mad_normalize(remaining_work)
    norm_wait = median_mad_normalize(ready_wait_time)
    norm_uncert = median_mad_normalize(uncertainty)
    tau = 0.16846524752787867
    ddl_gate = np.clip(slack / (tau + eps), 0.0, 1.0)
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-2.387705889365092 * (slack_pressure - 1.0)))
    critical_path_release = norm_rank * norm_work * (1.0 - ddl_gate)
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.654661174212836
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    coupled_rank = norm_rank * (1.0 + 2.4570308141922705 * rank_gate)
    uncert_gate = 1.0 / (1.0 + np.exp(-0.16846524752787867 * (norm_uncert - 0.1924117027001254)))
    robust_duration_penalty = norm_duration * norm_uncert * rank_gate * ddl_gate * uncert_gate
    wait_median = np.median(ready_wait_time) if N > 1 else ready_wait_time[0]
    normalized_wait = (ready_wait_time - wait_median) / (np.std(ready_wait_time) + eps) if N > 1 else 0.0
    load_like_metric = np.clip(normalized_wait, 0.0, 2.0)
    load_aware_gate = (load_like_metric > 0.9848189169759498).astype(float)
    wait_benefit = (1.0 - np.exp(-0.6300597221814751 * (ready_wait_time + eps))) * load_aware_gate
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    energy_slack_penalty = norm_energy * (1.0 + 2.654661174212836 * slack_pressure) * ddl_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_release, -2.0, 2.0) - np.clip(coupled_rank, -2.0, 2.0) - 1.6446751902221295 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.38599851501611104 * np.clip(robust_duration_penalty, -2.0, 2.0) + 0.6693331239685892 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 2.654661174212836 * np.clip(energy_slack_penalty, -2.0, 2.0) + 0.8080678854404846 * np.clip(norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
