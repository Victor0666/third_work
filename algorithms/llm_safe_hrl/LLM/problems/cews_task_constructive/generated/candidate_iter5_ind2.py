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
    Hybrid priority rule: combines Parent 2's urgency gating and risk-aware uncertainty boosting
    with Parent 1's robust IQR normalization and calibrated efficiency scaling.
    Key innovations:
      - Urgency gate remains primary dominance mechanism (hard DDL enforcement)
      - Criticality-energy coupling uses IQR-normalized upward_rank and median-scaled efficiency
        to preserve relative magnitude while suppressing outliers
      - Uncertainty boost is gated by both urgency AND slack magnitude, with linear ramp from
        median_slack to min_slack for precise risk severity modeling
      - Starvation guard enhanced: wait_boost activates only when (urgency < 0.65 AND slack >= 0)
        and is percentile-ranked for outlier robustness
      - All components fused via convex combination with weights tuned to prioritize
        deadline compliance first, then energy efficiency, then fairness
      - Final score bounded and fully finite-checked
    """
    eps = 1e-08
    # Ensure float arrays without modifying inputs
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

    # Robust IQR normalization function
    def normalize_robust(x):
        q75, q25 = np.percentile(x, [75, 25], method='midpoint')
        iqr = q75 - q25 + eps
        med = np.median(x)
        return (x - med) / iqr

    # Urgency: sigmoid centered at median slack, scaled by IQR for robustness
    median_slack = np.median(slack)
    iqr_slack = np.percentile(slack, 75) - np.percentile(slack, 25) + eps
    urgency = 1.0 / (1.0 + np.exp(-(slack - median_slack) / (iqr_slack + eps)))

    # Criticality-per-energy: upward_rank normalized robustly, efficiency scaled by median
    norm_upward = normalize_robust(upward_rank)
    efficiency = remaining_work / (min_incremental_energy + eps)
    median_eff = np.median(efficiency) + eps
    scaled_efficiency = np.clip(efficiency / median_eff, 0.1, 10.0)
    crit_per_energy = norm_upward * scaled_efficiency
    norm_crit_per_energy = normalize_robust(crit_per_energy)

    # Urgency-gated criticality term: activates only when urgency > 0.3
    urgency_gate_crit = np.where(urgency > 0.3, 1.0, 0.0)
    crit_term = (1.0 - np.clip(norm_crit_per_energy, -5.0, 5.0)) * urgency_gate_crit

    # Energy penalty: inverse urgency-scaled, using robustly normalized energy
    norm_energy = normalize_robust(min_incremental_energy)
    energy_penalty = np.clip(norm_energy, -5.0, 5.0) * (1.0 + 0.6 * urgency)

    # Uncertainty boost: linear ramp in slack space, gated by urgency
    min_slack = np.min(slack)
    slack_range = np.maximum(median_slack - min_slack, eps)
    uncertainty_risk_score = np.clip((median_slack - slack) / slack_range, 0.0, 1.0)
    uncertainty_boost = uncertainty * uncertainty_risk_score * np.where(urgency > 0.2, 1.0, 0.0)
    norm_uncertainty_boost = normalize_robust(uncertainty_boost)

    # Starvation guard: percentile-based wait boost, activated only for non-urgent & non-late tasks
    wait_gate = np.where((urgency < 0.65) & (slack >= 0), 1.0, 0.0)
    if N > 1:
        wait_percentile = np.argsort(np.argsort(ready_wait_time)) / (N - 1 + eps)
    else:
        wait_percentile = np.array([0.0])
    wait_boost = wait_percentile * wait_gate
    norm_wait_boost = normalize_robust(wait_boost)

    # Latency terms: execution and communication, robustly normalized
    norm_exec = normalize_robust(min_exec_time)
    norm_comm = normalize_robust(min_comm_time)
    latency_term = 0.5 * np.clip(norm_exec, -5.0, 5.0) + 0.5 * np.clip(norm_comm, -5.0, 5.0)

    # Base urgency dominates: lower urgency (i.e., more slack) → higher priority (smaller score)
    base_urgency = 1.0 - urgency

    # Convex combination with priority ordering: deadline > energy > latency > fairness > risk
    score = (
        0.42 * base_urgency +
        0.23 * energy_penalty +
        0.14 * latency_term +
        0.09 * crit_term +
        0.06 * (1.0 - np.clip(norm_wait_boost, -5.0, 5.0)) +
        0.04 * norm_uncertainty_boost +
        0.02 * (1.0 - np.clip(norm_crit_per_energy, -5.0, 5.0))
    )

    # Final safety: bound and sanitize
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    
    return score
