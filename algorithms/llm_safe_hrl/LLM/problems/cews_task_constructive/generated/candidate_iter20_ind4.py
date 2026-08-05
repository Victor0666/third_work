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
    v2 priority rule: Hybrid urgency-criticality-energy-fairness with robust risk-aware normalization.
    
    Key improvements:
    - Combines Parent 2's clipped inverse-linear slack sensitivity and latency-aware criticality gating
    - Adopts Parent 1's risk-adjusted starvation relief (only under high uncertainty AND positive slack)
    - Uses robust MAD-based normalization (Parent 1) for stability on small N, but with Parent 2's tighter clipping
    - Introduces dynamic energy-weighting decay: energy penalty attenuated by tanh(urgency) to avoid over-penalizing urgent tasks
    - Uncertainty coupling applied only to critical tasks via gated amplification (critical_gate * uncertainty)
    - All normalizations guard against zero-mad and NaN with epsilon fallbacks
    - Strict additive layering with decreasing weights preserves priority hierarchy
    """
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=np.float64).copy()
    min_comm_time = np.asarray(min_comm_time, dtype=np.float64).copy()
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=np.float64).copy()
    slack = np.asarray(slack, dtype=np.float64).copy()
    upward_rank = np.asarray(upward_rank, dtype=np.float64).copy()
    remaining_work = np.asarray(remaining_work, dtype=np.float64).copy()
    ready_wait_time = np.asarray(ready_wait_time, dtype=np.float64).copy()
    uncertainty = np.asarray(uncertainty, dtype=np.float64).copy()
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=np.float64)

    def robust_mad_norm(x):
        if x.size == 0:
            return np.zeros_like(x)
        x_clean = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        med = np.median(x_clean)
        mad = np.median(np.abs(x_clean - med))
        if mad < eps:
            return np.zeros_like(x_clean)
        z = (x_clean - med) / (mad + eps)
        return np.clip(z, -3.0, 3.0)

    # Urgency detection: hard deadline violation → highest priority
    is_urgent = (slack <= 0.0).astype(np.float64)

    # Duration and normalized slack for urgency sensitivity
    duration = min_exec_time + min_comm_time + eps
    abs_slack = np.abs(slack) + eps
    # Clipped inverse-linear slack sensitivity (Parent 2): strong signal near zero slack
    slack_sensitivity = np.clip(1.0 / abs_slack, 0.1, 20.0)
    # Latency-aware criticality gating: only penalize long-duration tasks when slack is tight
    rel_slack = np.divide(slack, duration, out=np.zeros_like(slack), where=duration != 0)
    rel_slack = np.nan_to_num(rel_slack, nan=0.0)
    tight_slack_mask = (rel_slack <= 0.25).astype(np.float64)
    duration_med = np.median(duration) + eps
    long_duration_mask = (duration >= duration_med).astype(np.float64)
    critical_gate = tight_slack_mask * long_duration_mask

    # Critical latency: duration * upward_rank, gated and normalized
    critical_latency_raw = duration * upward_rank
    norm_critical_latency = robust_mad_norm(critical_latency_raw)

    # Risk-normalized energy: penalize high incremental energy per effective duration
    effective_duration = duration * (1.0 + uncertainty + eps)
    risk_norm_energy = np.divide(min_incremental_energy, effective_duration,
                                 out=np.zeros_like(min_incremental_energy),
                                 where=effective_duration != 0)
    risk_norm_energy = np.nan_to_num(risk_norm_energy, nan=0.0, posinf=0.0, neginf=0.0)
    norm_energy = robust_mad_norm(risk_norm_energy)

    # Starvation relief: only activate under high uncertainty AND safe slack (Parent 1)
    high_uncertainty = (uncertainty > np.percentile(uncertainty, 80) + eps).astype(np.float64)
    safe_slack = (slack > 0.0).astype(np.float64)
    starvation_gate = high_uncertainty * safe_slack
    wait_per_duration = np.divide(ready_wait_time, duration,
                                  out=np.zeros_like(ready_wait_time),
                                  where=duration != 0)
    wait_per_duration = np.nan_to_num(wait_per_duration, nan=0.0, posinf=0.0, neginf=0.0)
    norm_wait_rel = robust_mad_norm(wait_per_duration)
    starvation_relief = starvation_gate * (0.3 * norm_wait_rel)

    # Dynamic energy weighting: reduce energy penalty for urgent tasks
    tanh_urgency = 0.5 * (1.0 - np.tanh(np.clip(slack / (duration_med + eps), -10.0, 10.0)))
    energy_weight = 0.2 + 0.8 * (1.0 - tanh_urgency)

    # Uncertainty-coupled criticality boost: amplify critical tasks under uncertainty
    unc_coupled_criticality = critical_gate * uncertainty
    norm_unc_coupled = robust_mad_norm(unc_coupled_criticality)

    # Work importance: larger remaining work gets slight bonus (lower score)
    norm_remaining_work = robust_mad_norm(remaining_work)
    work_bonus = -0.1 * norm_remaining_work

    # Base score construction with strict layering
    score = np.full(N, 0.0, dtype=np.float64)
    # Hard urgency override
    score = np.where(is_urgent, -1000000000000.0, score)
    # Layered contributions (decreasing weight ensures hierarchy)
    score = np.where(is_urgent, score, score + 0.35 * norm_critical_latency)
    score = np.where(is_urgent, score, score + 0.25 * energy_weight * norm_energy * critical_gate)
    score = np.where(is_urgent, score, score + 0.15 * starvation_relief)
    score = np.where(is_urgent, score, score + 0.12 * norm_unc_coupled)
    score = np.where(is_urgent, score, score + 0.13 * work_bonus)

    # Final clipping and sanitization
    score = np.clip(score, -1000000000000.0, 1000000000000.0)
    score = np.nan_to_num(score, nan=1000000000000.0, posinf=1000000000000.0, neginf=-1000000000000.0)
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
