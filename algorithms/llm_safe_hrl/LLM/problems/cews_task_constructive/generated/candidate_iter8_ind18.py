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
    Hybrid priority rule: combines v1's adaptive deadline sensitivity and slack-gated energy,
    with v0's orthogonal term design, robust IQR normalization, and explicit fairness via wait-ratio.
    Key innovations:
      - Unified urgency: adaptive sigmoid *and* hard deadline gating (urgency dominates when slack < 0)
      - Critical-energy density uses slack-gated scaling *and* critical-path boost (upward_rank × normalized remaining_work)
      - Fairness term replaces latency-ratio with robust wait-ratio (ready_wait_time / (exec+comm+eps)), clipped and normalized
      - Uncertainty penalty is additive, bounded, and activated only in high-risk/low-margin regime (0 < slack <= 2.5 & uncertainty > 0.03)
      - All normalizations use IQR-based robust_normalize with consistent eps and clipping to [-3,3]
      - Final score strictly finite, sanitized, and deterministic for identical inputs.
    """
    eps = 1e-08
    # Sanitize all inputs to avoid NaN/inf propagation
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=eps, posinf=1e6, neginf=eps)

    def robust_normalize(x):
        if x.size == 0:
            return np.zeros_like(x)
        if x.size == 1:
            return np.zeros_like(x)
        q25, q50, q75 = np.percentile(x, [25, 50, 75], method='midpoint')
        iqr = q75 - q25
        scale = np.where(iqr > eps, iqr + eps, 1.0)
        normed = (x - q50) / (scale + eps)
        return np.clip(normed, -3.0, 3.0)

    # === Urgency Term (dominant under deadline pressure) ===
    # Adaptive sigmoid steepness: steeper near zero, flatter for large |slack|
    tau_adapt = np.abs(np.median(slack)) + eps
    urgency_sigmoid = 1.0 / (1.0 + np.exp(-slack / (tau_adapt + 0.1)))
    # Hard gating: when slack < 0, urgency dominates → assign max priority (min score)
    urgency_raw = np.where(slack < 0.0, -100.0, urgency_sigmoid)
    norm_urgency = robust_normalize(urgency_raw)
    urgency_term = -4.5 * norm_urgency

    # === Critical-Energy Density Term (energy efficiency only when feasible) ===
    exec_effort = np.maximum(min_exec_time, eps)
    # Base density: importance per joule — upward_rank × work / energy
    base_density = (upward_rank * remaining_work) / (min_incremental_energy + eps)
    # Slack-gated: suppress energy savings when slack <= 0 (feasibility first)
    critical_energy_density = np.where(slack > 0.0, base_density, 0.0)
    # Critical-path boost: upward_rank scaled by normalized remaining_work relative to workflow median
    median_rw = np.median(remaining_work) + eps
    cp_boost = upward_rank * (remaining_work / median_rw)
    # Combine: weighted sum, then normalize
    combined_critical = 0.7 * critical_energy_density + 0.3 * cp_boost
    norm_critical = robust_normalize(combined_critical)
    critical_term = -2.0 * norm_critical

    # === Fairness Term (robust wait-ratio, not latency-ratio) ===
    duration_estimate = min_exec_time + min_comm_time + eps
    wait_ratio = np.clip(ready_wait_time / duration_estimate, 0.0, 10.0)
    norm_wait = robust_normalize(wait_ratio)
    # Clip fairness contribution to avoid overcompensation
    fairness_term = -0.4 * np.clip(norm_wait, -1.0, 1.5)

    # === Uncertainty Penalty (additive, bounded, conditional activation) ===
    # Only active in high-risk margin: 0 < slack <= 2.5 AND uncertainty > 0.03
    uncertainty_active = (slack > 0.0) & (slack <= 2.5) & (uncertainty > 0.03)
    # Sigmoidally weighted by proximity to zero slack
    unc_weight = 1.0 / (1.0 + np.exp(-(1.25 - slack)))
    uncertainty_penalty = np.where(uncertainty_active, uncertainty * unc_weight, 0.0)
    norm_unc = robust_normalize(uncertainty_penalty)
    uncertainty_term = 0.18 * norm_unc

    # === Aging Boost (for tasks stuck in ready set despite positive slack) ===
    # Activated when slack < 2.0 and wait time > 0.05s → prevents starvation
    aging_boost = np.where((slack < 2.0) & (ready_wait_time > 0.05),
                          np.tanh(1.0 * ready_wait_time / (np.abs(slack) + 0.3)),
                          0.0)
    aging_term = 0.12 * aging_boost

    # Aggregate all terms
    score = urgency_term + critical_term + fairness_term + uncertainty_term + aging_term

    # Final sanitization: ensure finite, bounded output
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)

    return score
