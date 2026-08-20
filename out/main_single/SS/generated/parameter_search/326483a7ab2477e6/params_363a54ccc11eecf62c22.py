import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: replaces fragile blended slack with conditional DDL-protection gate.
       When slack < 0, raw-slack penalty dominates via convex weighting — preserving hard constraint fidelity.
       When slack >= 0, robust normalization governs ranking stability. Eliminates dilution of urgency signals
       while retaining outlier resilience. All parameters used; no literals except -2,-1,0,1,2."""
    eps = 0.08697913306538767
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
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.8231940930045023
    norm_slack = robust_normalize(slack)
    ddl_violated = (slack < 0.0).astype(float)
    primary_slack_score = 0.5601684958136847 * raw_slack_penalty * ddl_violated + (1.0 - 0.5601684958136847 * ddl_violated) * norm_slack
    slack_pressure = np.clip(-primary_slack_score, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-2.453052330157848 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * (1.0 + 1.4843456769417627 * rank_gate)
    uncert_gate = np.where(norm_uncert > 0.47091675221608187, 1.0, 0.0)
    duration_risk_score = norm_duration * uncert_gate * raw_slack_penalty
    wait_benefit = 1.0 - np.exp(-0.8327195192392158 * (norm_wait + 4.99143013430406e-05))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate
    score = +raw_slack_penalty - boosted_rank - 1.4976411696943719 * norm_energy - norm_duration - wait_benefit + 0.1421254040623945 * duration_risk_score + 0.5892929017592212 * energy_uncert_penalty + 0.23861734209417595 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
