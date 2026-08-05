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
    Hybrid priority rule combining robustness, deadline gating, and risk-aware energy efficiency.
    
    Key improvements:
    - Uses IQR-based robust normalization (from Parent 2) for all features to resist outliers.
    - Implements slack-gated urgency with adaptive linear+soft transition (Parent 2) but adds slack-relative scaling.
    - Introduces *risk-weighted criticality-energy ratio*: upward_rank / (min_incremental_energy * (1 + uncertainty)), 
      normalized via IQR and capped to prevent explosion under high uncertainty.
    - Replaces sigmoid starvation bonus with *normalized waiting-time penalty* (bounded [0,0.3]) to avoid over-prioritization.
    - Adds *uncertainty-aware energy scaling*: min_incremental_energy is scaled by (1 + uncertainty) only when slack < 0.
    - All divisions protected by eps; NaN/inf handled via nan_to_num with finite bounds.
    - Ensures deterministic, finite output with strict shape (N,) and priority-by-min semantics.
    """
    eps = 1e-8
    # Convert inputs to float arrays without in-place modification
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

    # Robust IQR-based normalization function
    def robust_normalize(x):
        q1, q3 = np.quantile(x, [0.25, 0.75])
        iqr = q3 - q1 + eps
        center = np.median(x)
        return (x - center) / iqr

    # Deadline urgency: binary gating + linear penalty for negative slack, soft decay for positive slack
    # Use relative slack scaling: normalize slack by mean absolute slack to avoid scale drift
    abs_slack_mean = np.mean(np.abs(slack)) + eps
    slack_normalized = slack / abs_slack_mean
    urgency_mask = (slack <= 0.0).astype(float)
    urgency_linear = np.maximum(0.0, -slack_normalized) * 0.8
    urgency_soft = 1.0 / (1.0 + np.maximum(0.0, slack_normalized) * 0.2 + eps)
    deadline_urgency = urgency_mask * (1.0 + urgency_linear) + (1.0 - urgency_mask) * urgency_soft

    # Risk-weighted criticality-energy ratio: higher rank & lower energy → better ratio
    # Scale energy by uncertainty only when under deadline pressure (slack < 0)
    energy_risk_weighted = min_incremental_energy * (1.0 + np.where(slack < 0.0, uncertainty, 0.0))
    energy_safe = np.maximum(energy_risk_weighted, eps)
    crit_eff_ratio = upward_rank / energy_safe
    # Cap extreme ratios to prevent distortion from near-zero energy or zero rank
    crit_eff_ratio = np.clip(crit_eff_ratio, 1e-6, 1e6)
    crit_eff_norm = robust_normalize(crit_eff_ratio)

    # Starvation prevention: bounded normalized wait time (0 to 0.3), prevents dominance
    max_wait = np.max(ready_wait_time) + eps
    wait_penalty = np.clip(ready_wait_time / max_wait, 0.0, 1.0) * 0.3

    # Time cost: sqrt of exec + comm, robustly normalized
    time_cost = np.sqrt(np.maximum(min_exec_time, eps)) + np.sqrt(np.maximum(min_comm_time, eps))
    time_norm = robust_normalize(time_cost)

    # Work and uncertainty normalization
    work_norm = robust_normalize(remaining_work)
    unc_norm = robust_normalize(uncertainty)

    # Energy term: normalized raw energy, scaled up when slack < 0 (to penalize energy waste on risky tasks)
    energy_norm = robust_normalize(min_incremental_energy)
    energy_risk_penalty = np.where(slack < 0.0, energy_norm * 1.2, energy_norm * 0.8)

    # Final score: smaller = higher priority
    # Weights tuned to prioritize deadline urgency first, then criticality-efficiency, then fairness & risk
    score = (
        -3.0 * deadline_urgency           # Strongest pull: meet deadlines
        -1.5 * crit_eff_norm             # Favor high-impact low-risk energy tasks
        + 0.4 * time_norm                # Mild penalty for long-latency tasks
        + 0.3 * energy_risk_penalty      # Energy matters more under deadline stress
        + 0.25 * work_norm               # Prefer finishing large remaining work earlier
        + 0.15 * unc_norm                # Penalize high uncertainty when slack tight
        + wait_penalty                   # Anti-starvation (adds priority, so positive)
    )

    # Ensure finite, deterministic output
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    return score
