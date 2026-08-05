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
    v2 priority rule: Hybrid deadline-hardness + risk-isolated synergy + fairness-aware wait amplification.
    
    Key innovations:
      - Zero-tolerance urgency gating (like v1): slack <= 0 → urgent_bias = -1.0 ensures top priority
      - Synergy term from v0: upward_rank * duration / energy_density, robustly scaled to align latency-criticality-efficiency
      - Adaptive starvation suppression: only activates when slack > 0 AND ready_wait_time > quantile(0.75), scaled by normalized wait and slack proximity
      - Risk-coupled uncertainty: uses |slack|^{-1} clamping (v1) but multiplied by uncertainty_factor (v0-style 1+0.7*uncertainty)
      - Energy penalty refined with 90th-percentile threshold on criticality-weighted energy density (v1) + robust minmax norm
      - All normalizations use 1%-99% clipping + min-max; all divisions guarded; NaN/inf strictly sanitized
      - Final score hierarchy: urgency dominates (-1.0), then synergy (-0.8 weight), energy (0.22), latency (0.15), wait (0.10), uncertainty (0.08), work (0.04)
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

    def robust_minmax_norm(x):
        x_min = np.min(x)
        x_max = np.max(x)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        p01 = np.percentile(x, 1.0)
        p99 = np.percentile(x, 99.0)
        x_clipped = np.clip(x, p01, p99)
        x_min_c = np.min(x_clipped)
        x_max_c = np.max(x_clipped)
        if x_max_c - x_min_c < eps:
            return np.zeros_like(x)
        return (x_clipped - x_min_c) / (x_max_c - x_min_c + eps)

    # Urgency bias: hard zero-tolerance gating
    urgency_bias = np.where(slack <= 0, -1.0, 0.0)

    # Duration and synergy components (v0-inspired but normalized)
    duration = min_exec_time + min_comm_time + eps
    energy_density = min_incremental_energy / (duration + eps)
    synergy_raw = upward_rank * duration / (energy_density + eps)
    synergy_score = -robust_minmax_norm(synergy_raw)

    # Energy penalty: v1-style criticality-normalized thresholding
    eff_per_rank = energy_density / (upward_rank + eps)
    threshold_eff_per_rank = np.percentile(eff_per_rank, 90.0) + eps
    energy_penalty_mask = (eff_per_rank > threshold_eff_per_rank).astype(float)
    norm_energy_density = robust_minmax_norm(energy_density)
    energy_penalty = norm_energy_density * energy_penalty_mask

    # Latency: execution + communication, weighted by criticality & uncertainty (v1+v0 hybrid)
    comm_weight = 1.0 + 0.6 * robust_minmax_norm(upward_rank) + 0.4 * robust_minmax_norm(uncertainty)
    weighted_comm = min_comm_time * comm_weight
    latency_raw = min_exec_time + weighted_comm + eps
    norm_latency = robust_minmax_norm(latency_raw)

    # Fairness: starvation-aware wait penalty (v0 adaptive + v1 masking)
    starvation_mask = (slack > 0).astype(float)
    wait_thresh = np.quantile(ready_wait_time, 0.75) if N > 1 else np.max(ready_wait_time)
    wait_boost_raw = np.where(ready_wait_time > wait_thresh, (ready_wait_time - wait_thresh) / (np.std(ready_wait_time) + eps), 0.0)
    wait_penalty = starvation_mask * robust_minmax_norm(wait_boost_raw)

    # Risk-coupled uncertainty: v1's |slack|^{-1} scaling × v0's uncertainty factor
    slack_abs = np.abs(slack) + 1.0
    slack_scale_factor = np.clip(1.0 / slack_abs, 0.1, 10.0)
    uncertainty_factor = 1.0 + 0.7 * np.clip(uncertainty, 0.0, 2.0)
    uncertainty_boost = uncertainty_factor * slack_scale_factor
    norm_uncertainty_boost = robust_minmax_norm(uncertainty_boost)

    # Remaining work normalized
    norm_remaining_work = robust_minmax_norm(remaining_work)

    # Final weighted score: urgency dominates; synergy strongly negative (higher priority)
    score = (
        1.0 + urgency_bias +
        0.8 * synergy_score +
        0.22 * energy_penalty +
        0.15 * norm_latency +
        0.10 * wait_penalty +
        0.08 * norm_uncertainty_boost +
        0.04 * norm_remaining_work
    )

    # Sanitize and clamp
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)

    # Ensure shape
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
