import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: merges Parent 2's strict DDL-dominant energy disable gate and logistic starvation relief
       with Parent 1's robust slack-margin-ratio-based urgency detection and successor-release interaction.
       Novel improvement: dual-gated energy modulation — combines both safe_slack_mask AND uncertainty-aware activation
       to avoid over-suppression under high uncertainty even with moderate slack; ensures risk-aware energy tradeoffs
       only when both feasibility and uncertainty conditions are jointly satisfied."""
    eps = 0.0001256117120257214
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
    ddl_urgent = np.where((slack <= 0.0) | (slack_margin_ratio < 0.03021716477571774), 1.0, 0.0)
    safe_slack_mask = norm_slack > 0.03021716477571774
    energy_enabled = np.where(safe_slack_mask, 0.0, 1.0)
    energy_modulation_gate = np.where((norm_uncert > 0.688976802685687) & (ddl_urgent == 1.0), 1.0, 0.0)
    slack_pressure_norm = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-6.530361473619324 * (slack_pressure_norm - 1.0)))
    successor_release_impact = norm_work * norm_rank
    wait_benefit = 1.0 - np.exp(-0.24127375362363956 * (norm_wait + 1.4470250880907637e-05))
    duration_risk_score = norm_duration * norm_uncert * ddl_urgent
    energy_uncert_penalty = norm_energy * norm_uncert * energy_modulation_gate * energy_enabled
    score = +np.clip(norm_slack, -2.0, 2.0) + np.clip(norm_slack ** 3.4648686453128628, -2.0, 2.0) * ddl_urgent - np.clip(norm_rank * (1.0 + 2.7788019678888722 * rank_gate * ddl_urgent), -2.0, 2.0) - np.clip(successor_release_impact, -2.0, 2.0) - np.clip(1.6995968179388141 * norm_energy * energy_modulation_gate * energy_enabled, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(0.26212227421062784 * duration_risk_score, -2.0, 2.0) + np.clip(0.9665913477280276 * energy_uncert_penalty, -2.0, 2.0) + np.clip(1.9995075212269562 * norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
