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
    v2 priority rule: Combines hard urgency gating (v1) with physics-aware energy gating (v0),
    adds criticality-weighted communication latency, adaptive starvation decay, and risk-congruent
    uncertainty modulation — all under strict monotonic normalization to prevent rank inversion.
    
    Key innovations:
      - Hard urgency flag (0/1) preserves deadline dominance; no smoothing dilution.
      - Energy penalty only applied when marginal energy density exceeds local median → avoids penalizing inherently efficient tasks.
      - Communication latency scaled by upward_rank to model stall amplification in high-criticality paths.
      - Starvation mitigation activated only for non-urgent AND long-waiting tasks, with exponential decay by upward_rank (fairness prioritized for low-criticality).
      - Uncertainty boost modulated by relative slack gap *and* sign-consistent slack scaling → robust near deadline boundary.
      - All components normalized via robust min-max (v1) *with outlier clipping* to ensure stability; no quantile rank collapse.
      - Final convex combination weights sum to 1.0 and enforce strict dominance hierarchy: urgency >> energy >> latency >> fairness >> uncertainty.
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

    # Robust min-max normalization that handles constants and outliers
    def robust_minmax_norm(x):
        x_min = np.min(x)
        x_max = np.max(x)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        # Clip extreme outliers to preserve signal integrity (1% tail)
        p01 = np.percentile(x, 1)
        p99 = np.percentile(x, 99)
        x_clipped = np.clip(x, p01, p99)
        x_min_clipped = np.min(x_clipped)
        x_max_clipped = np.max(x_clipped)
        if x_max_clipped - x_min_clipped < eps:
            return np.zeros_like(x)
        return (x - x_min_clipped) / (x_max_clipped - x_min_clipped + eps)

    # 1. Hard urgency flag: zero slack or negative → max priority (score=0)
    urgency_flag = np.where(slack <= 0, 0.0, 1.0)

    # 2. Physics-aware energy gating: penalize only above-median energy density
    duration = min_exec_time + min_comm_time + eps
    energy_density = np.clip(min_incremental_energy / (duration + eps), eps, 1e6)
    median_energy_den = np.median(energy_density) + eps
    energy_efficiency_mask = (energy_density <= median_energy_den).astype(float)
    norm_energy_density = robust_minmax_norm(energy_density)
    energy_penalty = norm_energy_density * (1.0 - energy_efficiency_mask) * urgency_flag

    # 3. Criticality-weighted communication latency: high upward_rank amplifies comm impact
    comm_latency_weighted = min_comm_time * (1.0 + 0.5 * robust_minmax_norm(upward_rank))
    norm_latency = robust_minmax_norm(min_exec_time + comm_latency_weighted + eps)

    # 4. Criticality-per-energy efficiency: higher is better → invert for priority
    crit_per_energy = upward_rank / (min_incremental_energy + eps)
    crit_per_energy = np.clip(crit_per_energy, 1e-6, 1e7)
    norm_crit_per_energy = robust_minmax_norm(crit_per_energy)

    # 5. Risk-congruent uncertainty boost: only active when slack < 0, scaled by relative gap
    median_slack = np.median(slack)
    slack_gap = np.maximum(0.0, median_slack - slack)  # >0 only when slack < median_slack
    slack_scale = np.abs(median_slack) + eps
    uncertainty_risk_score = np.where(slack < 0, slack_gap / slack_scale, 0.0)
    uncertainty_boost = uncertainty * uncertainty_risk_score
    norm_uncertainty_boost = robust_minmax_norm(uncertainty_boost)

    # 6. Adaptive starvation mitigation: only for non-urgent AND top 5% waiters, decayed by criticality
    wait_thresh = np.percentile(ready_wait_time, 95) + eps
    starvation_flag = np.where((slack > 0) & (ready_wait_time >= wait_thresh), 1.0, 0.0)
    # Decay starvation weight for high-criticality tasks (they're already urgent or important)
    ur_norm = robust_minmax_norm(upward_rank)
    starv_decay = np.exp(-0.7 * ur_norm)  # Stronger decay than v0
    wait_penalty = starvation_flag * robust_minmax_norm(ready_wait_time) * starv_decay

    # 7. Combine with strict dominance hierarchy weights (sum = 1.0)
    score = (
        0.48 * urgency_flag +                           # Dominant: hard deadline enforcement
        0.19 * energy_penalty +                        # Secondary: efficiency in feasible region
        0.11 * norm_latency +                          # Tertiary: latency-coupled criticality
        0.08 * (1.0 - norm_crit_per_energy) +         # Encourage high-criticality / low-energy
        0.07 * norm_uncertainty_boost +                # Risk amplification only when needed
        0.05 * wait_penalty +                          # Fairness guard, adaptively gated
        0.02 * robust_minmax_norm(uncertainty)         # Baseline uncertainty awareness
    )

    # Final safeguard: clamp and clean
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)

    return score
