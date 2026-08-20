import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: replaces unstable successor-release coupling with bounded slack-conditioned energy decay;
       restores robust MAD-based normalization (more stable for small N); introduces rank-uncertainty coupling only under DDL feasibility;
       clips all intermediate terms to [-2,2] and enforces monotonic term-wise contribution; eliminates unbounded mask products."""
    eps = 0.00015948305220837516
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def robust_normalize(x):
        x_abs = np.abs(x)
        if N == 1:
            center = x_abs[0]
            spread = eps
        else:
            center = np.median(x_abs)
            spread = np.median(np.abs(x_abs - center))
        return (x_abs - center) / (spread + eps)
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.7680921431581256
    norm_slack_penalty = robust_normalize(raw_slack_penalty)
    coupled_slack = np.clip(norm_slack, -1.0, 1.0)
    rank_slack_coupling = 1.0 + 0.5224759114278483 * coupled_slack
    boosted_rank = norm_rank * rank_slack_coupling
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-4.07060696095015 * (slack_pressure - 1.0)))
    boosted_rank = boosted_rank * (1.0 + 0.5224759114278483 * rank_gate)
    ddl_gate = 1.0 / (1.0 + np.exp(-9.713812971032574 * slack))
    energy_decay_factor = np.exp(-0.9982417407192771 * np.maximum(slack, 0.0))
    energy_preference = norm_energy * ddl_gate * energy_decay_factor
    rank_uncert_coupling = norm_rank * norm_uncert * ddl_gate
    rank_uncert_boost = 0.5265261958270545 * np.clip(rank_uncert_coupling, -2.0, 2.0)
    uncert_gate = 1.0 / (1.0 + np.exp(-9.713812971032574 * (norm_uncert - 0.7071711456450267)))
    duration_risk_score = norm_duration * norm_uncert * uncert_gate * ddl_gate
    wait_benefit = 1.0 - np.exp(-0.2897558059352433 * norm_wait)
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 0.11616215389022773 * np.clip(energy_preference, -2.0, 2.0) - 0.5283578811369642 * np.clip(norm_duration * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(duration_risk_score, -2.0, 2.0) + rank_uncert_boost + 1.178793361410471 * np.clip(norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
