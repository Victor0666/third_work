import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: combines Parent 2's binary DDL gate and sign-preserving normalization with Parent 1's slack-pressure gating insight,
       enhanced by a novel adaptive slack-mode switch and strengthened successor-release interaction.
       Introduces slack_sensitivity_threshold to smoothly transition between raw-penalty and normalized regimes — improving stability near zero slack.
       Replaces fragile sigmoid with piecewise linear activation for better CMA-ES convergence and interpretability.
       All 12 parameters are used; no numeric literals beyond -2,-1,0,1,2."""
    eps = 0.030559385960763318
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
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.8674989151890045
    slack_mode_weight = np.clip((0.2964392824412163 - slack) / (0.2964392824412163 + eps), 0.0, 1.0)
    primary_slack_score = slack_mode_weight * raw_slack_penalty + (1.0 - slack_mode_weight) * norm_slack
    ddl_urgent = np.where((slack <= 0.0) | (slack <= 0.2964392824412163) & (uncertainty >= 0.6305312658406502), 1.0, 0.0)
    successor_release_boost = norm_rank * norm_work * 1.256968058491192
    protected_release_score = successor_release_boost * ddl_urgent
    energy_uncert_penalty = norm_energy * norm_uncert * np.where((norm_energy > 0.0) & (norm_uncert > 0.6305312658406502), 1.0, 0.0)
    wait_benefit = np.clip(0.0006940454824521199 * (ready_wait_time + 5.8458643810879526e-08), 0.0, 1.0)
    duration_risk_score = norm_duration * np.where((slack <= 0.2964392824412163) & (uncertainty >= 0.6305312658406502), 1.0, 0.0)
    score = +primary_slack_score - 1.4030224444416604 * protected_release_score - 0.9329726826496852 * norm_energy - norm_duration - wait_benefit + 0.05119626889200186 * duration_risk_score + 0.05207379825550151 * energy_uncert_penalty + 0.36568702475869597 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
