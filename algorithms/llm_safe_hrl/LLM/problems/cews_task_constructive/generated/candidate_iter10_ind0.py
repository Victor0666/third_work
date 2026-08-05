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
    Mutated priority rule emphasizing:
    - Hard-thresholded urgency gating via 30th-percentile relative slack
    - Risk-adaptive energy scaling using clipped exponential decay of slack
    - Latency-criticality synergy: (upward_rank * task_duration) / energy, normalized robustly
    - Slack-gated starvation control: wait boosting only when slack > slack_30pct
    - Uncertainty-weighted time penalty to penalize high-risk long-duration tasks
    - All features normalized via clipped z-score (mean ± 3*std) for stability
    """
    eps = 1e-8
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

    # Robust z-score normalization with clipping: mean ± 3*std bounds
    def robust_zscore(x):
        mu = np.mean(x)
        std = np.std(x, ddof=0) + eps
        z = (x - mu) / std
        return np.clip(z, -3.0, 3.0)

    # Task intrinsic duration and risk-adjusted time cost
    task_duration = np.maximum(min_exec_time + min_comm_time, eps)
    risk_weighted_duration = task_duration * (1.0 + uncertainty)

    # Hard-thresholded urgency gating: use 30th-percentile relative slack as gate
    rel_slack = slack / task_duration
    slack_30pct = np.quantile(rel_slack, 0.3) if N > 1 else np.min(rel_slack)
    is_urgent = (rel_slack <= slack_30pct).astype(float)
    
    # Urgency score: linear ramp for critical tasks, soft decay for non-critical
    urgency_score = np.where(
        is_urgent,
        1.0 + np.maximum(0.0, -rel_slack) * 0.8,
        1.0 / (1.0 + np.maximum(0.0, rel_slack - slack_30pct) * 0.5 + eps)
    )

    # Risk-adaptive energy scaling: exponential slack penalty for energy term
    # alpha = clip(1.0 - 0.3 * slack, 0.5, 2.5) → higher penalty when slack negative
    alpha = np.clip(1.0 - 0.3 * slack, 0.5, 2.5)
    energy_risk_scaled = min_incremental_energy * np.power(1.0 + uncertainty, alpha)
    energy_safe = np.maximum(energy_risk_scaled, eps)

    # Latency-criticality synergy: (upward_rank * risk_weighted_duration) / energy_safe
    # Reflects "importance per joule per second" — prioritizes high-impact, fast, efficient tasks
    synergy_num = upward_rank * risk_weighted_duration
    latency_criticality_synergy = np.divide(synergy_num, energy_safe, out=np.full(N, 1e-6), where=energy_safe != 0)
    latency_criticality_synergy = np.clip(latency_criticality_synergy, 1e-6, 1e6)

    # Slack-gated starvation control: only activate wait boost for non-urgent tasks
    wait_boost = np.where(
        rel_slack > slack_30pct,
        np.clip(ready_wait_time / (np.maximum(np.median(ready_wait_time), eps) + eps), 0.0, 0.3),
        0.0
    )

    # Uncertainty-weighted time penalty: discourage high-risk long tasks
    time_uncertainty_penalty = uncertainty * risk_weighted_duration
    time_uncertainty_penalty = robust_zscore(time_uncertainty_penalty)

    # Normalize all components
    urgency_norm = robust_zscore(urgency_score)
    synergy_norm = robust_zscore(latency_criticality_synergy)
    energy_norm = robust_zscore(energy_safe)
    work_norm = robust_zscore(remaining_work)
    unc_norm = robust_zscore(uncertainty)
    wait_norm = robust_zscore(wait_boost)

    # Final score: smaller = higher priority
    # Dominant terms: urgency (negative weight), synergy (negative), wait (positive)
    # Secondary: energy (positive), uncertainty (positive), work (neutral)
    score = (
        -2.5 * urgency_norm           # Strongest pull toward urgent tasks
        -1.8 * synergy_norm           # Favor high-impact, low-energy, fast execution
        +0.4 * energy_norm            # Mild penalty for high marginal energy
        +0.3 * unc_norm               # Penalize high uncertainty
        +0.2 * time_uncertainty_penalty  # Extra penalty for risky long durations
        +0.25 * wait_norm             # Small fairness boost for long-waiting non-urgent tasks
        +0.1 * work_norm              # Neutral work-aware term (avoids starvation of large jobs)
    )

    # Ensure finite output
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    return score
