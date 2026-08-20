import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: combines Parent 2's binary DDL gate and sign-preserving normalization with Parent 1's slack-pressure gating insight,
       enhanced by a novel adaptive slack-mode switch and strengthened successor-release interaction.
       Introduces slack_sensitivity_threshold to smoothly transition between raw-penalty and normalized regimes — improving stability near zero slack.
       Replaces fragile sigmoid with piecewise linear activation for better CMA-ES convergence and interpretability.
       All 12 parameters are used; no numeric literals beyond -2,-1,0,1,2."""
    eps = 0.009513005639213182
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
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.9148708992052272
    slack_mode_weight = np.clip((0.41775822636705906 - slack) / (0.41775822636705906 + eps), 0.0, 1.0)
    primary_slack_score = slack_mode_weight * raw_slack_penalty + (1.0 - slack_mode_weight) * norm_slack
    ddl_urgent = np.where((slack <= 0.0) | (slack <= 0.41775822636705906) & (uncertainty >= 0.679842499808627), 1.0, 0.0)
    successor_release_boost = norm_rank * norm_work * 0.9662154111422198
    protected_release_score = successor_release_boost * ddl_urgent
    energy_uncert_penalty = norm_energy * norm_uncert * np.where((norm_energy > 0.0) & (norm_uncert > 0.679842499808627), 1.0, 0.0)
    wait_benefit = np.clip(0.3302385205107943 * (ready_wait_time + 7.46842774383158e-08), 0.0, 1.0)
    duration_risk_score = norm_duration * np.where((slack <= 0.41775822636705906) & (uncertainty >= 0.679842499808627), 1.0, 0.0)
    score = +primary_slack_score - 1.1133726110758735 * protected_release_score - 0.8266588705776674 * norm_energy - norm_duration - wait_benefit + 0.09967611080825603 * duration_risk_score + 0.3037350664363205 * energy_uncert_penalty + 0.47343592903946147 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
