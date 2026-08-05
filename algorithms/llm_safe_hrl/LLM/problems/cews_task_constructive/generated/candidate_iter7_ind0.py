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
    - Slack-gated urgency via adaptive sigmoid threshold (30th percentile slack)
    - Risk-adjusted energy density scaled by *normalized duration uncertainty* rather than raw uncertainty
    - Bounded min-max normalization (not IQR) for cross-seed stability and outlier resilience
    - Wait boost gated by *both* slack and remaining work to prevent starvation without compromising DDL
    - Criticality-energy tradeoff using *inverse energy density per unit work*, not raw ratio
    - Explicit penalty term for high uncertainty *only when slack is tight* (avoid over-penalizing robust tasks)
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

    # Robust min-max normalization with epsilon-clipped range and centering
    def robust_minmax(x):
        x_min, x_max = np.min(x), np.max(x)
        rng = x_max - x_min + eps
        centered = x - x_min
        return np.clip(centered / rng, 0.0, 1.0)

    # Task intrinsic duration: execution + communication (lower = faster)
    task_duration = min_exec_time + min_comm_time + eps

    # Relative slack: normalized by task duration; negative = urgent
    rel_slack = slack / task_duration

    # Adaptive urgency gate: sigmoid activated only below 30th percentile relative slack
    # Ensures hard DDL enforcement while preserving smoothness for feasible tasks
    slack_threshold = np.quantile(rel_slack, 0.3) if N > 1 else np.median(rel_slack)
    urgency_gate = 1.0 / (1.0 + np.exp(5.0 * (rel_slack - slack_threshold)))

    # Lateness risk penalty: exponential ramp for negative rel_slack, capped
    lateness_penalty = np.where(rel_slack < 0.0,
                               np.clip(-rel_slack * 2.0, 0.0, 5.0),
                               0.0)

    # Risk-adjusted energy density: normalize energy by work and scale by *duration uncertainty*
    # Use coefficient of variation (std/mean) of task_duration — more stable than raw uncertainty
    dur_uncertainty = np.where(task_duration > eps,
                              uncertainty / task_duration,
                              0.0)
    # Energy per unit work, scaled by duration uncertainty — higher score = worse efficiency
    energy_per_work = np.maximum(min_incremental_energy / (remaining_work + eps), eps)
    energy_density_risk = energy_per_work * (1.0 + np.clip(dur_uncertainty, 0.0, 2.0))

    # Criticality-weighted energy density: upward_rank modulates energy cost importance
    # High rank → energy matters more; low rank → energy less penalized
    crit_weighted_energy = energy_density_risk * (1.0 + upward_rank / (np.median(upward_rank + eps) + eps))

    # Normalize components with robust min-max (not IQR) for deterministic cross-run stability
    norm_urgency = robust_minmax(urgency_gate)
    norm_lateness = robust_minmax(lateness_penalty)
    norm_energy = robust_minmax(crit_weighted_energy)
    norm_upward = robust_minmax(upward_rank)
    norm_work = robust_minmax(remaining_work)
    norm_wait = robust_minmax(ready_wait_time)

    # Wait boost: only active for non-urgent tasks (rel_slack > -0.1) AND non-trivial work
    wait_boost = np.where((rel_slack > -0.1) & (remaining_work > np.quantile(remaining_work, 0.25) + eps),
                         norm_wait * 0.3 * (1.0 - norm_urgency),
                         0.0)

    # Tight-slack uncertainty penalty: only penalize uncertainty when rel_slack <= 0.1
    tight_slack_mask = (rel_slack <= 0.1).astype(float)
    unc_penalty = tight_slack_mask * robust_minmax(dur_uncertainty) * 0.4

    # Final score: smaller = better
    # Prioritize urgency first, then penalize inefficient energy use, then reward criticality & work,
    # add fairness (wait boost), and penalize uncertainty under pressure
    score = (
        -2.8 * norm_urgency                    # Strongest weight: deadline feasibility
        + 1.5 * norm_lateness                  # Penalty for imminent lateness
        + 1.2 * norm_energy                    # Penalize high-risk energy density
        - 0.9 * norm_upward                    # Reward high criticality (inverted)
        - 0.6 * norm_work                      # Reward large remaining work (inverted)
        + wait_boost                           # Fairness: non-urgent waiting tasks
        + unc_penalty                          # Uncertainty penalty only when tight
    )

    # Ensure finite output, no NaN/inf
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    
    return score
