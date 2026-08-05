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
    Hybrid priority rule v2: Combines Parent 2's urgency gating and risk-aware uncertainty scaling
    with Parent 1's work-aware urgency modulation and robust percentile-based normalization.
    Key innovations:
      - Urgency remains primary gate via sigmoid, but augmented with work-impact scaling for late tasks
      - Criticality-energy term uses clipped energy density (not raw energy) to avoid distortion near zero
      - Uncertainty boost now jointly gated by urgency magnitude *and* slack deficit severity
      - Starvation penalty applies only to non-urgent, non-late tasks, scaled by log-wait and criticality
      - All features normalized via percentile scaling (monotonic, outlier-robust) with explicit bounds
      - Final score enforces strict priority hierarchy: urgency > criticality-efficiency > fairness > latency
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

    # Compute robust duration and energy density (Parent 1 style, eps-protected)
    duration = min_exec_time + min_comm_time + eps
    energy_density = np.clip(min_incremental_energy / (duration + eps), eps, 1e6)

    # Sigmoid urgency: primary gate (Parent 2 style)
    median_slack = np.median(slack)
    iqr_slack = np.percentile(slack, 75) - np.percentile(slack, 25) + eps
    urgency = 1.0 / (1.0 + np.exp(-(slack - median_slack) / (iqr_slack + eps)))

    # Work-aware urgency modulation for late tasks (Parent 1 enhancement)
    slack_deficit = np.maximum(0.0, -slack)
    median_rw = np.median(remaining_work) + eps
    work_impact_factor = np.where(slack <= 0.0, np.clip(remaining_work / median_rw, 0.3, 3.0), 1.0)
    # Augmented urgency: preserves dominance of high-impact late tasks
    augmented_urgency = np.clip(urgency + 0.3 * (1.0 - urgency) * work_impact_factor * (slack_deficit > 0), 0.0, 1.0)

    # Percentile normalization (Parent 1 style: monotonic, robust to outliers)
    def percentile_scale(x):
        if len(x) == 1:
            return np.array([0.5])
        ranks = np.argsort(np.argsort(x)) + 1.0
        return np.clip(ranks / (len(x) + 1.0), 0.0, 1.0)

    dur_norm = percentile_scale(duration)
    energy_den_norm = percentile_scale(energy_density)
    ur_norm = percentile_scale(upward_rank)
    rw_norm = percentile_scale(remaining_work)
    unc_norm = percentile_scale(uncertainty)
    wait_norm = percentile_scale(np.log1p(ready_wait_time))

    # Criticality-efficiency term: upward_rank per unit energy_density, gated by urgency
    crit_eff_score = upward_rank / (energy_density + eps)
    crit_eff_norm = percentile_scale(np.clip(crit_eff_score, eps, 1e7))
    # Activate only when urgency > 0.4 — stronger gate than Parent 2's 0.3 to reduce noise
    crit_gate = np.where(augmented_urgency > 0.4, 1.0, 0.0)
    crit_term = (1.0 - crit_eff_norm) * crit_gate

    # Energy penalty: inverse urgency-scaled (Parent 2) but applied to energy_density, not raw energy
    energy_penalty = energy_den_norm * (1.0 + 0.7 * augmented_urgency)

    # Risk-aware uncertainty boost: joint gate on urgency magnitude AND slack deficit severity
    # Uses linear ramp from median_slack down to min_slack (Parent 2 insight), scaled by uncertainty
    slack_range = np.maximum(np.abs(np.min(slack) - median_slack), eps)
    uncertainty_risk_score = np.clip((median_slack - slack) / slack_range, 0.0, 1.0)
    uncertainty_boost = uncertainty * uncertainty_risk_score * (1.0 - augmented_urgency)  # dampen for urgent tasks
    unc_boost_norm = percentile_scale(uncertainty_boost)

    # Starvation guard: only for non-urgent (urgency < 0.6) and non-late (slack >= 0) tasks
    wait_gate = np.where((augmented_urgency < 0.6) & (slack >= 0), 1.0, 0.0)
    wait_penalty = wait_norm * wait_gate * (1.0 - ur_norm)  # reward waiting only for low-criticality tasks

    # Latency term: normalized exec + comm, weighted equally
    latency_term = 0.5 * dur_norm

    # Final convex combination with strict hierarchy weights
    # Urgency dominates (0.5), then energy penalty (0.2), criticality (0.15), latency (0.08), fairness (0.05), risk boost (0.02)
    score = (
        0.50 * (1.0 - augmented_urgency) +
        0.20 * energy_penalty +
        0.15 * crit_term +
        0.08 * latency_term +
        0.05 * wait_penalty +
        0.02 * (1.0 - unc_boost_norm)
    )

    # Ensure finite output, deterministic shape, and bounded values
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)
    return score
