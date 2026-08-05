import numpy as np

def get_task_priority_v2(
    min_exec_time,
    min_comm_time,
    min_incremental_energy,
    slack,
    upward_rank,
    remaining_work,
    ready_wait_time,
    uncertainty
):
    """
    Hybrid priority rule combining v1's robustness with v0's strict DDL enforcement:
    - Uses adaptive sigmoid urgency gating (30th percentile relative slack) for smooth deadline dominance
    - Applies *hard urgency override* only for critically negative slack (lateness > 10% of median task duration)
    - Normalizes uncertainty relative to task duration (v1) but boosts it *only when slack is borderline* (v0 insight)
    - Replaces percentile-based wait boost with log-scaled monotonic fairness guard (v0) gated by slack and work
    - Criticality-energy term uses inverse energy density per work, weighted by upward rank (v1), normalized robustly
    - Energy penalty is slack-aware: intensifies as slack shrinks toward zero, not just median
    - All normalizations use bounded min-max with fallbacks; all ops protected against div-by-zero/Nan/inf
    """
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)

    def robust_minmax(x):
        x_min, x_max = np.min(x), np.max(x)
        rng = x_max - x_min + eps
        return np.clip((x - x_min) / rng, 0.0, 1.0)

    # Compute base metrics
    task_duration = min_exec_time + min_comm_time + eps
    rel_slack = slack / task_duration
    median_task_dur = np.median(task_duration)
    
    # Adaptive urgency: sigmoid on relative slack (v1) + hard override for severe lateness (v0)
    slack_threshold = np.quantile(rel_slack, 0.3) if N > 1 else np.median(rel_slack)
    urgency_sigmoid = 1.0 / (1.0 + np.exp(5.0 * (rel_slack - slack_threshold)))
    # Hard override: tasks with slack < -0.1 * median_task_dur get max urgency priority
    severe_lateness = (slack < -0.1 * median_task_dur).astype(float)
    urgency = np.where(severe_lateness, 1.0, urgency_sigmoid)
    
    # Lateness penalty: only active when slack is negative, capped
    lateness_penalty = np.where(slack < 0.0, np.clip(-slack / (median_task_dur + eps), 0.0, 5.0), 0.0)
    
    # Uncertainty normalized by duration (v1), then boosted only for borderline slack (v0 insight)
    dur_uncertainty = np.where(task_duration > eps, uncertainty / task_duration, 0.0)
    # Borderline slack: 0.0 <= rel_slack <= 0.2 → highest uncertainty impact zone
    borderline_mask = ((rel_slack >= 0.0) & (rel_slack <= 0.2)).astype(float)
    uncertainty_boost = dur_uncertainty * borderline_mask
    
    # Criticality-energy term: energy density per work, weighted by upward rank (v1)
    energy_per_work = np.maximum(min_incremental_energy / (remaining_work + eps), eps)
    rank_scale = 1.0 + upward_rank / (np.median(upward_rank + eps) + eps)
    crit_weighted_energy = energy_per_work * rank_scale
    
    # Fairness guard: log-scaled wait time (v0), gated by slack and work to prevent starvation
    wait_guard = np.log1p(ready_wait_time) / np.log1p(np.max(ready_wait_time) + eps)
    # Gate: only activate fairness boost when slack is non-critical AND work is substantial
    wait_gate = (rel_slack > -0.05).astype(float) * (remaining_work > np.quantile(remaining_work, 0.25) + eps).astype(float)
    wait_boost = wait_guard * wait_gate * 0.3
    
    # Energy penalty: slack-aware scaling — strongest near zero slack
    slack_abs_norm = np.abs(slack) / (np.abs(np.median(slack)) + eps)
    energy_penalty_weight = np.clip(1.0 - slack_abs_norm, 0.0, 1.0)
    energy_penalty = robust_minmax(min_incremental_energy) * energy_penalty_weight
    
    # Normalize components
    norm_urgency = robust_minmax(urgency)
    norm_lateness = robust_minmax(lateness_penalty)
    norm_energy = robust_minmax(crit_weighted_energy)
    norm_uncertainty = robust_minmax(uncertainty_boost)
    norm_wait = wait_boost  # already scaled [0, 0.3]
    
    # Final weighted score: smaller = higher priority
    # Weights emphasize DDL compliance (urgency/lateness), then energy efficiency, then fairness & risk
    score = (
        -2.5 * norm_urgency           # high urgency → low score
        + 1.4 * norm_lateness         # lateness penalty → higher score
        + 1.1 * energy_penalty       # energy inefficiency → higher score
        + 0.05 * norm_uncertainty    # uncertainty in borderline zone → mild penalty
        + 0.25 * norm_wait           # fairness boost → lowers score for long-waiting tasks
        + 0.03 * robust_minmax(upward_rank)  # minor criticality bonus (lower score for high rank)
    )
    
    # Clip and sanitize
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    
    return score
