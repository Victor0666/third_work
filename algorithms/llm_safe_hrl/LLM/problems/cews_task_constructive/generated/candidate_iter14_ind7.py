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
    v2 priority rule: Hybrid design combining Parent 2's zero-tolerance urgency and percentile energy filtering
    with Parent 1's robust MAD normalization, work-normalized communication pressure, and starvation-aware wait scaling.
    Key innovations:
      - Urgency remains binary (-1.0 for slack <= 0) but adds *lateness magnitude amplification* for highly negative slack
      - Replaces percentile thresholding with *criticality-weighted MAD-based outlier suppression* for energy penalty
      - Integrates Parent 1's comm-to-work ratio with Parent 2's slack-conditioned scaling (1.5x under deadline violation)
      - Uses *adaptive starvation weight*: linearly increases with slack surplus to prioritize fairness when deadlines are safe
      - Applies unified robust normalization using clipped MAD (more stable than min-max for skewed distributions)
      - Introduces *slack-aware uncertainty gating*: uncertainty only modulates priority when slack > 0; zero otherwise
      - Ensures all components are strictly bounded and division-safe
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

    def robust_normalize_mad(x):
        center = np.median(x)
        mad = np.median(np.abs(x - center)) + eps
        normed = (x - center) / mad
        return np.clip(normed, -8.0, 8.0)

    # Zero-tolerance urgency with lateness magnitude boost: more negative slack → higher priority
    urgency_bias = np.where(slack <= 0.0, -1.0 - np.clip(-slack / (np.maximum(np.abs(slack).max(), eps)), 0.0, 2.0), 0.0)

    # Criticality-normalized energy density: energy per unit duration per upward rank
    duration = min_exec_time + min_comm_time + eps
    energy_density = min_incremental_energy / duration
    eff_per_rank = energy_density / (upward_rank + eps)
    
    # MAD-based robust energy penalty: penalize outliers in eff_per_rank, not all high values
    eff_norm = robust_normalize_mad(eff_per_rank)
    energy_penalty_mask = (eff_norm > 2.0).astype(float)  # Strong outliers only
    energy_penalty = np.abs(eff_norm) * energy_penalty_mask

    # Communication pressure: high-comm/low-work tasks prioritized under tight slack
    comm_to_work_ratio = min_comm_time / (remaining_work + eps)
    comm_pressure = np.where(slack < 0.0, comm_to_work_ratio * 1.5, comm_to_work_ratio * 0.4)
    comm_norm = robust_normalize_mad(comm_pressure)

    # Starvation control: linearly increasing weight when slack > 0 (fairness when safe), zero when urgent
    max_slack_safe = np.maximum(np.max(slack[slack > 0.0], initial=eps), eps)
    starvation_weight = np.where(slack > 0.0, np.clip(slack / max_slack_safe, 0.0, 1.0) * 0.3, 0.0)
    norm_wait_time = robust_normalize_mad(ready_wait_time)
    wait_penalty = starvation_weight * norm_wait_time

    # Slack-aware uncertainty modulation: only active when slack > 0; scaled by |slack|^{-1} clamped
    slack_abs = np.abs(slack) + 1.0
    slack_scale_factor = np.clip(1.0 / slack_abs, 0.1, 10.0)
    uncertainty_modulated = np.where(slack > 0.0, uncertainty * slack_scale_factor, 0.0)
    norm_uncertainty_mod = robust_normalize_mad(uncertainty_modulated)

    # Baseline uncertainty and remaining work (mild regularization)
    norm_uncertainty = robust_normalize_mad(uncertainty)
    norm_remaining_work = robust_normalize_mad(remaining_work)

    # Final score: urgency dominates; others weighted for balance and stability
    # Weights tuned to preserve hierarchy: urgency > energy > comm > wait > uncertainty_mod > uncertainty > work
    score = (
        1.0 + 
        urgency_bias + 
        0.25 * energy_penalty + 
        0.18 * comm_norm + 
        0.12 * wait_penalty + 
        0.09 * norm_uncertainty_mod + 
        0.06 * norm_uncertainty + 
        0.04 * norm_remaining_work
    )

    # Final safeguard: ensure finite, bounded output
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)
    
    return score
