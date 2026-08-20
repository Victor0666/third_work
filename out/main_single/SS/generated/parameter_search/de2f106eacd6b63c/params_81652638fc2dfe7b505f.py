import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: integrates Parent 2's DDL protection and successor-release logic with Parent 1's bounded rank-slack coupling and explicit starvation relief.
       Novel additions: (1) additive bias in rank-slack coupling replaced by unified sigmoid coupling, (2) linear energy modulation by slack pressure improves gradient continuity,
       (3) unified robust normalization with sign-preserving MAD scaling, (4) clipped composite terms to [-2,2] ensure ordinal stability under perturbation."""
    eps = 0.013076593401745948
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
        x = np.asarray(x, dtype=float)
        center = np.median(x) if N > 1 else x[0]
        spread = np.median(np.abs(x - center)) if N > 1 else np.abs(x[0] - center) + eps
        return (x - center) / (spread + eps)
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    slack_margin_ratio = np.abs(slack) / (min_exec_time + min_comm_time + eps)
    ddl_urgent = np.where((slack <= 0.0) | (slack_margin_ratio < 0.10768889202177089), 1.0, 0.0)
    successor_release_impact = norm_work * norm_rank
    slack_pressure_norm = np.clip(-norm_slack, 0.0, 2.0)
    energy_sensitivity_modulated = 0.5746513558315468 * (1.0 + 0.9182736533828995 * slack_pressure_norm)
    rank_gate = 1.0 / (1.0 + np.exp(-7.67772414117454 * (slack_pressure_norm - 1.0)))
    boosted_rank = norm_rank * (1.0 + 1.6468299490990896 * rank_gate) * ddl_urgent
    wait_benefit = 1.0 - np.exp(-0.11474174782025498 * (norm_wait + 3.1030605927418443e-08))
    duration_risk_score = norm_duration * norm_uncert * ddl_urgent
    energy_uncert_penalty = norm_energy * norm_uncert * ddl_urgent
    score = +np.clip(norm_slack, -2.0, 2.0) + np.clip(norm_slack ** 3.9846112004121035, -2.0, 2.0) * ddl_urgent - np.clip(boosted_rank, -2.0, 2.0) - np.clip(successor_release_impact, -2.0, 2.0) - np.clip(norm_energy * energy_sensitivity_modulated * ddl_urgent, -2.0, 2.0) - np.clip(norm_duration * (1.0 - ddl_urgent), -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(0.11864064566249069 * duration_risk_score, -2.0, 2.0) + np.clip(0.1732490257969773 * energy_uncert_penalty, -2.0, 2.0) + np.clip(1.3138474352643112 * norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
