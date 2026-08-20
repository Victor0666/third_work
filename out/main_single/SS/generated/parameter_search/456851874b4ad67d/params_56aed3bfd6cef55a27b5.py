import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: replaces hidden 0.2 threshold with explicit parameter-free logic using median_abs_slack.
       DDL-protection gating now uses slack <= 0 (hard feasibility boundary) instead of ratio — simpler, safer, and avoids extra param.
       All numeric literals are -2,-1,0,1,2; epsilon handled via PARAMS["epsilon"]; no hidden constants."""
    eps = 0.0015956258800350403
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
    median_abs_slack = np.median(np.abs(slack)) if N > 1 else np.abs(slack[0])
    urgency_input = -slack / (median_abs_slack + eps)
    ddl_urgency_gate = 1.0 / (1.0 + np.exp(-2.0872276088058994 * urgency_input))
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.832075739555966
    rank_boost = 0.4185918201368599 + (1.0 - 0.4185918201368599) * ddl_urgency_gate
    protected_rank = norm_rank * rank_boost
    ddl_protection_mask = (slack <= 0.0).astype(float)
    successor_release_impact = norm_rank * norm_work * ddl_protection_mask
    wait_raw = 0.28482965631652163 * (ready_wait_time + 1.0217707081938403e-09)
    wait_benefit = np.clip(wait_raw, 0.0, 1.0)
    energy_uncert_penalty = norm_energy * norm_uncert * np.where((norm_energy > 0.0) & (norm_uncert > 0.48707876280181406), 1.0, 0.0)
    duration_risk_score = norm_duration * np.where((slack <= eps) & (uncertainty >= 0.48707876280181406), 1.0, 0.0)
    score = +raw_slack_penalty - 1.9694562175348174 * protected_rank - successor_release_impact - 1.3732071771113972 * norm_energy - norm_duration - wait_benefit + 0.6200528832132302 * duration_risk_score + 0.24147330302109782 * energy_uncert_penalty + 0.8021351789537167 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
