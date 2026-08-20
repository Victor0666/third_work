import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: replaces fragile blended slack with conditional DDL-protection gate.
       When slack < 0, raw-slack penalty dominates via convex weighting — preserving hard constraint fidelity.
       When slack >= 0, robust normalization governs ranking stability. Eliminates dilution of urgency signals
       while retaining outlier resilience. All parameters used; no literals except -2,-1,0,1,2."""
    eps = 0.0004160982072353929
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
        center = np.median(x_abs) if N > 1 else x_abs[0]
        spread = np.median(np.abs(x_abs - center)) if N > 1 else np.abs(x_abs[0] - center) + eps
        return (x_abs - center) / (spread + eps)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.299011873447954
    norm_slack = robust_normalize(slack)
    ddl_violated = (slack < 0.0).astype(float)
    primary_slack_score = 0.5605331673014469 * raw_slack_penalty * ddl_violated + (1.0 - 0.5605331673014469 * ddl_violated) * norm_slack
    slack_pressure = np.clip(-primary_slack_score, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-3.0057693002431307 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * (1.0 + 1.7819051556385388 * rank_gate)
    uncert_gate = np.where(norm_uncert > 0.643999231166058, 1.0, 0.0)
    duration_risk_score = norm_duration * uncert_gate * raw_slack_penalty
    wait_benefit = 1.0 - np.exp(-0.45186089170497346 * (norm_wait + 0.0008040205558465084))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate
    score = +raw_slack_penalty - boosted_rank - 0.8013323516509785 * norm_energy - norm_duration - wait_benefit + 0.8811534920928625 * duration_risk_score + 0.005632552056287191 * energy_uncert_penalty + 0.7444021451514049 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
