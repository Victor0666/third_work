import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: replaces hard binary energy disable with smooth, slack-aligned energy tradeoff ramp.
       Key structural change: introduces `energy_activation_slack_threshold` and `slack_energy_tradeoff_width`
       to enable continuous, differentiable energy weighting — preserving DDL dominance while allowing marginal
       energy optimization when slack is slightly positive (e.g., 0 < slack < 0.3×duration), improving total energy
       without violating deadlines. Removes redundant `wait_saturation_offset`. Uses raw slack (not normalized)
       for activation logic to preserve physical interpretability and robustness to distribution shifts.
       Adds `uncertainty_robustness_weight` to explicitly penalize high-uncertainty long-duration tasks under urgency."""
    eps = 0.060088825779726594
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
    energy_ramp_center = -0.2738401563317916
    energy_ramp_width = 0.1604883976285716
    energy_activation = np.clip((slack - energy_ramp_center) / (energy_ramp_width + eps), 0.0, 1.0)
    slack_margin_ratio = np.abs(slack) / (min_exec_time + min_comm_time + eps)
    ddl_urgent = np.where((slack <= 0.0) | (slack_margin_ratio < 0.1604883976285716), 1.0, 0.0)
    slack_pressure_norm = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-3.1821916682241507 * (slack_pressure_norm - 1.0)))
    successor_release_impact = norm_work * norm_rank
    wait_benefit = 1.0 - np.exp(-0.09805524336084115 * norm_wait)
    duration_risk_score = norm_duration * norm_uncert * ddl_urgent
    energy_uncert_penalty = norm_energy * norm_uncert * energy_activation
    score = +np.clip(norm_slack, -2.0, 2.0) + np.clip(norm_slack ** 2.501254029515498, -2.0, 2.0) * ddl_urgent - np.clip(norm_rank * (1.0 + 0.7779689696430983 * rank_gate * ddl_urgent), -2.0, 2.0) - np.clip(successor_release_impact, -2.0, 2.0) - np.clip(0.9295962154039378 * norm_energy * energy_activation, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(1.2258367441144753 * duration_risk_score, -2.0, 2.0) + np.clip(0.9709734990730893 * energy_uncert_penalty, -2.0, 2.0) + np.clip(0.29417503670524003 * norm_work, -2.0, 2.0) + np.clip(0.035976959522874 * norm_uncert * ddl_urgent, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
