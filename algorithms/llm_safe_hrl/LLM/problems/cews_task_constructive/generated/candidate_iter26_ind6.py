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
    v2 priority rule: Hybrid urgency-criticality-energy hierarchy with:
      - Robust adaptive slack gating (Parent 2) + dynamic uncertainty scaling (Parent 1)
      - Criticality-weighted slack sensitivity (Parent 2) + slack-driven energy penalty attenuation (Parent 1)
      - Work-density fairness under global margin (Parent 2) + starvation-robust wait gating (Parent 1)
      - Unified trimmed-MAD normalization with N=1 and zero-MAD fallbacks
      - Strict priority layering: Urgency > Criticality×SlackSens > GatedEnergy > WorkBonus > Fairness
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
        x_clean = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if N == 1:
            return np.zeros_like(x_clean)
        med = np.median(x_clean)
        abs_dev = np.abs(x_clean - med)
        mad = np.median(abs_dev)
        if mad < eps:
            # Fallback to minmax on clipped finite values
            x_finite = x_clean[np.isfinite(x_clean)]
            if x_finite.size == 0:
                return np.zeros_like(x_clean)
            p05 = np.percentile(x_finite, 5.0, method='midpoint')
            p95 = np.percentile(x_finite, 95.0, method='midpoint')
            x_clipped = np.clip(x_clean, p05, p95)
            x_min = np.min(x_clipped)
            x_max = np.max(x_clipped)
            if x_max - x_min < eps:
                return np.zeros_like(x_clean)
            return (x_clipped - x_min) / (x_max - x_min + eps)
        z = (x_clean - med) / (mad + eps)
        return np.clip(z, -3.0, 3.0)

    # Core timing and risk metrics
    duration = min_exec_time + min_comm_time + eps
    rel_slack = np.divide(slack, duration, out=np.zeros_like(slack), where=duration != 0)
    rel_slack = np.nan_to_num(rel_slack, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Adaptive tightness threshold (Parent 2)
    rel_slack_med = np.median(rel_slack)
    rel_slack_mad = np.median(np.abs(rel_slack - rel_slack_med))
    adaptive_tight_threshold = rel_slack_med + 0.5 * max(rel_slack_mad, eps)
    tight_slack_mask = (rel_slack <= adaptive_tight_threshold).astype(np.float64)
    
    # Urgency gating: hard deadline violation triggers top priority
    is_urgent = (slack <= eps).astype(np.float64)
    
    # Critical latency: weighted by upward rank (Parent 2)
    critical_latency_raw = duration * upward_rank
    norm_critical_latency = robust_mad_norm(critical_latency_raw)
    
    # Slack sensitivity: smooth tanh-based peak at slack=0, scaled by criticality (Parent 2)
    slack_norm = np.divide(slack, duration + eps, out=np.zeros_like(slack), where=duration != 0)
    slack_norm = np.nan_to_num(slack_norm, nan=0.0)
    # Bounded, symmetric sensitivity peaking at slack=0
    slack_sensitivity = 0.5 * (np.tanh(slack_norm / 0.5) - np.tanh((slack_norm - 2.0) / 0.5))
    critical_slack_sensitivity = upward_rank * slack_sensitivity
    
    # Uncertainty scaling: dynamic alpha (0.7) + sigmoid regularization (Parent 1 & 2 hybrid)
    uncertainty_alpha = 0.7
    scaled_uncertainty = np.power(np.maximum(uncertainty, eps), uncertainty_alpha)
    unc_sig = 1.0 / (1.0 + np.exp(-uncertainty))
    effective_duration = duration * (1.0 + scaled_uncertainty * unc_sig + eps)
    
    # Risk-adjusted energy density with slack-driven attenuation (Parent 1)
    risk_energy_density = np.divide(min_incremental_energy, effective_duration, 
                                   out=np.zeros_like(min_incremental_energy), 
                                   where=effective_duration != 0)
    risk_energy_density = np.nan_to_num(risk_energy_density, nan=0.0, posinf=0.0, neginf=0.0)
    # Attenuation mask: decays smoothly for slack > 0.1s (Parent 1)
    slack_positive = np.maximum(slack, 0.0)
    energy_penalty_mask = np.clip(1.0 - (slack_positive - 0.1) / 0.2, 0.0, 1.0)
    # Combine with criticality gating
    energy_gate = tight_slack_mask * energy_penalty_mask
    
    norm_energy_density = robust_mad_norm(risk_energy_density)
    
    # Work-density fairness: bonus for high remaining_work only under safe global margin (Parent 2)
    global_slack_med = np.median(slack) + eps
    safe_global_margin = (slack > global_slack_med).astype(np.float64)
    norm_remaining_work = robust_mad_norm(remaining_work)
    work_density_bonus = -0.09 * norm_remaining_work * safe_global_margin
    
    # Starvation-robust fairness: wait-per-work with minimum threshold (Parent 1)
    wait_per_work = np.divide(ready_wait_time, remaining_work + eps, 
                             out=np.zeros_like(ready_wait_time), 
                             where=remaining_work + eps != 0)
    wait_per_work = np.nan_to_num(wait_per_work, nan=0.0, posinf=0.0, neginf=0.0)
    wait_per_work = np.maximum(wait_per_work, 0.001)  # Prevent ultra-short task starvation
    wpw_finite = wait_per_work[np.isfinite(wait_per_work)]
    work_threshold = np.percentile(wpw_finite, 10.0) + eps if len(wpw_finite) > 0 else eps
    wait_gate = (wait_per_work >= work_threshold).astype(np.float64)
    norm_wait_per_work = robust_mad_norm(wait_per_work)
    wait_penalty = (1.0 - is_urgent) * norm_wait_per_work * wait_gate
    
    # Priority layering: strict ordering ensures feasibility-first behavior
    score = np.full(N, 0.0, dtype=np.float64)
    # Layer 1: Urgency override
    score = np.where(is_urgent, -1000000000000.0, score)
    # Layer 2: Criticality × Slack Sensitivity (highest weight for deadline-critical tasks)
    score = np.where(is_urgent, score, score + 0.46 * norm_critical_latency * critical_slack_sensitivity)
    # Layer 3: Gated energy density (only when slack tight and penalty active)
    score = np.where(is_urgent, score, score + 0.25 * norm_energy_density * energy_gate)
    # Layer 4: Work density bonus (under global safety margin)
    score = np.where(is_urgent, score, score + work_density_bonus)
    # Layer 5: Fairness penalty (wait pressure only when not urgent)
    score = np.where(is_urgent, score, score + 0.14 * wait_penalty)
    
    # Final clipping and NaN protection
    score = np.clip(score, -1000000000000.0, 1000000000000.0)
    score = np.nan_to_num(score, nan=1000000000000.0, posinf=1000000000000.0, neginf=-1000000000000.0)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
