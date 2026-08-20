import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: combines Parent 2's binary DDL gate and sign-preserving normalization with Parent 1's slack-pressure gating insight,
       enhanced by a novel adaptive slack-mode switch and strengthened successor-release interaction.
       Introduces slack_sensitivity_threshold to smoothly transition between raw-penalty and normalized regimes — improving stability near zero slack.
       Replaces fragile sigmoid with piecewise linear activation for better CMA-ES convergence and interpretability.
       All 12 parameters are used; no numeric literals beyond -2,-1,0,1,2."""
    eps = 5.683915972847415e-05
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
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.9692473028892987
    slack_mode_weight = np.clip((0.33118334967284746 - slack) / (0.33118334967284746 + eps), 0.0, 1.0)
    primary_slack_score = slack_mode_weight * raw_slack_penalty + (1.0 - slack_mode_weight) * norm_slack
    ddl_urgent = np.where((slack <= 0.0) | (slack <= 0.33118334967284746) & (uncertainty >= 0.10662563520909246), 1.0, 0.0)
    successor_release_boost = norm_rank * norm_work * 1.1959173231389093
    protected_release_score = successor_release_boost * ddl_urgent
    energy_uncert_penalty = norm_energy * norm_uncert * np.where((norm_energy > 0.0) & (norm_uncert > 0.10662563520909246), 1.0, 0.0)
    wait_benefit = np.clip(0.15639113047596814 * (ready_wait_time + 3.119740556321552e-08), 0.0, 1.0)
    duration_risk_score = norm_duration * np.where((slack <= 0.33118334967284746) & (uncertainty >= 0.10662563520909246), 1.0, 0.0)
    score = +primary_slack_score - 1.738371616779987 * protected_release_score - 1.69423400635292 * norm_energy - norm_duration - wait_benefit + 0.2309496383932057 * duration_risk_score + 0.28101005841876237 * energy_uncert_penalty + 0.5594434918205482 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
