import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule: replaces linear rank coupling with successor-release interaction;
       adds conditional DDL protection gate; uses median-MAD per-feature normalization;
       enforces hard deadline feasibility via binary urgency gating before composite scoring."""
    eps = 0.0174364101298244
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
    ddl_urgent_mask = (slack <= 0.0).astype(float)
    successor_release_score = norm_rank * norm_work * ddl_urgent_mask
    slack_gap = np.maximum(-slack, 0.0)
    ddl_protection_gate = 1.0 / (1.0 + np.exp(-7.672476378820684 * (slack_gap - eps)))
    boosted_rank = norm_rank * (1.0 + 1.6214809583838696 * ddl_protection_gate)
    energy_uncert_penalty = norm_energy * norm_uncert * np.where(norm_uncert > 0.5003103072543837, 1.0, 0.0)
    duration_risk_score = norm_duration * norm_uncert * ddl_protection_gate
    wait_benefit = 1.0 - np.exp(-0.032727727495131906 * (norm_wait + 1.05863806759821e-09))
    score = +robust_normalize(np.maximum(-slack, 0.0) ** 1.3906486587131122) - successor_release_score - boosted_rank - 0.30355965001111873 * norm_energy - wait_benefit + 0.007992439707644244 * duration_risk_score + 0.822721335128882 * energy_uncert_penalty + 1.7054120212719952 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
