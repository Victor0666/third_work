import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: restores uncertainty-aware energy gating, replaces soft modulation with hard DDL-dominant energy disabling,
       and strengthens deadline feasibility via strict slack-based conditional logic.
       Key structural change: introduces 'safe_slack_energy_disable_threshold' to fully zero out energy terms when slack is sufficiently positive,
       enforcing strict priority ordering where deadline feasibility always dominates energy minimization."""
    eps = 0.0066540928038742635
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
    ddl_urgent = np.where((slack <= 0.0) | (slack_margin_ratio < 0.6413978636498817), 1.0, 0.0)
    safe_slack_mask = norm_slack > 0.6413978636498817
    energy_enabled = np.where(safe_slack_mask, 0.0, 1.0)
    successor_release_impact = norm_work * norm_rank
    energy_modulation_gate = np.where((norm_uncert > 0.2850583405336597) & (ddl_urgent == 1.0), 1.0, 0.0)
    slack_pressure_norm = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-2.74867482967456 * (slack_pressure_norm - 1.0)))
    boosted_rank = norm_rank * (1.0 + 0.9367297604197168 * rank_gate * ddl_urgent)
    wait_benefit = 1.0 - np.exp(-0.8892868418235393 * (norm_wait + 4.192004088136645e-06))
    duration_risk_score = norm_duration * norm_uncert * ddl_urgent
    energy_uncert_penalty = norm_energy * norm_uncert * energy_modulation_gate * energy_enabled
    score = +np.clip(norm_slack, -2.0, 2.0) + np.clip(norm_slack ** 1.256597067252815, -2.0, 2.0) * ddl_urgent - np.clip(boosted_rank, -2.0, 2.0) - np.clip(successor_release_impact, -2.0, 2.0) - np.clip(norm_energy * 0.8400341662651104 * energy_modulation_gate * energy_enabled, -2.0, 2.0) - np.clip(norm_duration * (1.0 - ddl_urgent), -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(0.2568895858348938 * duration_risk_score, -2.0, 2.0) + np.clip(0.5832573322657771 * energy_uncert_penalty, -2.0, 2.0) + np.clip(0.8180148666469828 * norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
