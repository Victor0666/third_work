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
    Hybrid priority rule combining deadline dominance (v1) with robust normalization and risk-aware energy efficiency (v0).
    Key innovations:
      - Urgency remains primary gate (sigmoid-based, slack-centered), but uses IQR-stabilized scaling for noise resilience
      - Criticality-energy term is urgency-gated *and* normalized via robust IQR (not min-max) to avoid outlier distortion
      - Energy dominance term (energy/sqrt(duration)) retained from v0, robustly normalized and scaled by urgency to prioritize low-risk efficient tasks when safe, high-impact ones when urgent
      - Uncertainty boost now uses *signed slack distance* (median_slack - slack) clipped to [0,1], multiplied by uncertainty — captures severity of tight-deadline risk
      - Starvation guard activated only when (urgency < 0.6 AND slack >= 0) to prevent late-task reward while ensuring fairness
      - All terms bounded and fused via convex combination with fixed weights summing to 1.0; final score clamped and nan-cleaned
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

    def robust_normalize(x):
        q1, q3 = np.quantile(x, [0.25, 0.75], method='midpoint')
        iqr = q3 - q1
        if iqr < eps:
            std_val = np.std(x) + eps
            return (x - np.mean(x)) / std_val
        return (x - np.median(x)) / (iqr + eps)

    def robust_minmax_norm(x):
        x_min, x_max = np.min(x), np.max(x)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x - x_min) / (x_max - x_min + eps)

    # Urgency: sigmoid centered at median slack, scaled by IQR for stability
    median_slack = np.median(slack)
    iqr_slack = np.percentile(slack, 75) - np.percentile(slack, 25) + eps
    urgency = 1.0 / (1.0 + np.exp(-(slack - median_slack) / (iqr_slack + eps)))

    # Base urgency penalty: higher urgency → lower score (more priority)
    base_urgency_penalty = 1.0 - urgency

    # Criticality-per-energy: normalized robustly, gated by urgency > 0.3
    crit_per_energy = upward_rank / (min_incremental_energy + eps)
    norm_crit_per_energy = robust_normalize(crit_per_energy)
    urgency_gate_crit = np.where(urgency > 0.3, 1.0, 0.0)
    crit_term = (1.0 - robust_minmax_norm(norm_crit_per_energy)) * urgency_gate_crit

    # Energy dominance: min_incremental_energy / sqrt(duration), prioritizes efficient low-risk tasks
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_dominance = min_incremental_energy / np.sqrt(duration + eps)
    norm_energy_dom = robust_normalize(np.clip(energy_dominance, 1e-06, 1e6))
    # Scale energy dominance by (1 - urgency) so it dominates when safe, recedes when urgent
    energy_dom_term = (1.0 - urgency) * robust_minmax_norm(norm_energy_dom)

    # Energy penalty: high energy penalized more under urgency
    norm_energy = robust_minmax_norm(min_incremental_energy)
    energy_penalty = norm_energy * (1.0 + 0.5 * urgency)

    # Uncertainty boost: strongest when slack is tight *and* uncertainty high
    slack_range = np.maximum(np.abs(np.min(slack) - median_slack), eps)
    uncertainty_risk_score = np.clip((median_slack - slack) / slack_range, 0.0, 1.0)
    uncertainty_boost = uncertainty * uncertainty_risk_score
    norm_uncertainty_boost = robust_minmax_norm(uncertainty_boost)

    # Starvation guard: activates only for non-urgent, on-time tasks with long wait
    wait_gate = np.where((urgency < 0.6) & (slack >= 0), 1.0, 0.0)
    wait_boost = ready_wait_time * wait_gate
    norm_wait_boost = robust_minmax_norm(wait_boost)

    # Latency term: weighted exec + comm, normalized
    norm_exec = robust_minmax_norm(min_exec_time)
    norm_comm = robust_minmax_norm(min_comm_time)
    latency_term = 0.5 * norm_exec + 0.5 * norm_comm

    # Final convex combination (weights sum to 1.0)
    score = (
        0.40 * base_urgency_penalty +
        0.20 * energy_penalty +
        0.12 * latency_term +
        0.10 * crit_term +
        0.08 * energy_dom_term +
        0.05 * (1.0 - norm_wait_boost) +
        0.03 * norm_uncertainty_boost +
        0.02 * (1.0 - robust_minmax_norm(norm_crit_per_energy))
    )

    # Ensure finite output: clamp and sanitize
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)

    return score
