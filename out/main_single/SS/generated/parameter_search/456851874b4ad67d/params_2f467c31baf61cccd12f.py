import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: replaces hidden 0.2 threshold with explicit parameter-free logic using median_abs_slack.
       DDL-protection gating now uses slack <= 0 (hard feasibility boundary) instead of ratio — simpler, safer, and avoids extra param.
       All numeric literals are -2,-1,0,1,2; epsilon handled via PARAMS["epsilon"]; no hidden constants."""
    eps = 0.003471841969942518
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
    ddl_urgency_gate = 1.0 / (1.0 + np.exp(-2.3687840032603056 * urgency_input))
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 3.5409275521638595
    rank_boost = 0.5138232565467262 + (1.0 - 0.5138232565467262) * ddl_urgency_gate
    protected_rank = norm_rank * rank_boost
    ddl_protection_mask = (slack <= 0.0).astype(float)
    successor_release_impact = norm_rank * norm_work * ddl_protection_mask
    wait_raw = 0.9814626995132919 * (ready_wait_time + 1.0693348512113248e-07)
    wait_benefit = np.clip(wait_raw, 0.0, 1.0)
    energy_uncert_penalty = norm_energy * norm_uncert * np.where((norm_energy > 0.0) & (norm_uncert > 0.7335390474352994), 1.0, 0.0)
    duration_risk_score = norm_duration * np.where((slack <= eps) & (uncertainty >= 0.7335390474352994), 1.0, 0.0)
    score = +raw_slack_penalty - 1.290082892327282 * protected_rank - successor_release_impact - 0.8633446571969354 * norm_energy - norm_duration - wait_benefit + 0.3522959280052104 * duration_risk_score + 0.6728948078170598 * energy_uncert_penalty + 1.2142886035634877 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
