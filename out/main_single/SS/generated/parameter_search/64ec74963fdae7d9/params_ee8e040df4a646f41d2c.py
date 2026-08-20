import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: MAD-based normalization; sign-aware slack handling;
       simplified single-sigmoid urgency gate; no redundant comm coupling; all 12 params used."""
    eps = 0.005581329757365286
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
    slack_sign = np.sign(slack)
    norm_slack_signed = slack_sign * np.clip(np.abs(norm_slack), 0.0, 2.0)
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.5561565791592393
    norm_slack_penalty = robust_normalize(raw_slack_penalty)
    slack_pressure = np.clip(-slack, 0.0, np.inf)
    if N == 1:
        urgency_center = slack_pressure[0]
        urgency_spread = eps
    else:
        urgency_center = np.median(slack_pressure)
        urgency_spread = np.median(np.abs(slack_pressure - urgency_center)) + eps
    norm_urgency = (slack_pressure - urgency_center) / (urgency_spread + eps)
    rank_gate = 1.0 / (1.0 + np.exp(-5.334376089583481 * (norm_urgency - 1.0)))
    boosted_rank = norm_rank * (1.0 + 2.389690675093399 * rank_gate)
    ddl_gate = 1.0 / (1.0 + np.exp(-5.311585151256671 * slack))
    uncert_gate = 1.0 / (1.0 + np.exp(-5.311585151256671 * (norm_uncert - 0.014480302167095421)))
    duration_risk_score = norm_duration * uncert_gate * rank_gate * ddl_gate
    wait_benefit = 1.0 - np.exp(-0.6393966033728531 * (norm_wait + 0.00016776146776168897))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    tight_slack_mask = (slack > 0) & (slack < 0.014480302167095421 * (np.max(np.abs(slack)) + eps))
    energy_slack_penalty = norm_energy * (1.0 + 2.5561565791592393 * np.where(tight_slack_mask, 1.0, 0.0)) * ddl_gate
    duration_preference = 0.7810736835951287 * norm_duration * ddl_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 1.0994581939470958 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - duration_preference - np.clip(wait_benefit, -2.0, 2.0) + 0.7810736835951287 * np.clip(duration_risk_score, -2.0, 2.0) + 0.5831781413798018 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 2.5561565791592393 * np.clip(energy_slack_penalty, -2.0, 2.0) + 0.5314576496962178 * np.clip(norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
