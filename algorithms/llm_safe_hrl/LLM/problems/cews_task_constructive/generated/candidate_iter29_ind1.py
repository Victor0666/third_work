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
    v2 priority rule: Hybrid deadline-feasibility + risk-energy optimization with:
      - Hard zero-slack gating (strict urgency dominance)
      - Criticality×SlackSens decoupled from energy layer to preserve feasibility focus
      - Inverted energy score only when slack > 0 (avoids risky low-energy assignments)
      - Robust uncertainty-normalized work consolidation for load/reliability balance
      - Starvation-aware relative age normalized by critical-path length
      - Unified robust MAD-based normalization with N=1 safety and strict finite bounds
      - Explicit lateness dominance via unclipped linear penalty for slack <= 0
      - Quadratic risk-duration scaling for energy density under uncertainty
    """
    eps = 1e-08
    # Sanitize inputs: convert, copy, and nan/inf handling
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    slack = np.nan_to_num(np.asarray(slack, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=np.float64)
    
    def robust_mad_norm(x):
        """MAD-based robust normalization with N=1 safety and finite clipping"""
        x_clean = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if N == 1:
            return np.zeros_like(x_clean)
        med = np.median(x_clean)
        abs_dev = np.abs(x_clean - med)
        mad = np.median(abs_dev)
        if mad < eps:
            # Fallback: percentile-based scaling when MAD near zero
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
    
    # Core duration and derived metrics
    duration = min_exec_time + min_comm_time + eps
    is_urgent = (slack <= eps).astype(np.float64)
    is_safe = (slack > eps).astype(np.float64)
    
    # Lateness dominance: unclipped linear penalty for violated deadlines
    slack_deficit = np.maximum(-slack, 0.0)
    norm_deficit = np.divide(slack_deficit, duration, out=np.zeros_like(slack_deficit), where=duration != 0)
    norm_deficit = np.nan_to_num(norm_deficit, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Criticality × Slack sensitivity: prioritize high-criticality late tasks
    slack_norm = np.divide(slack, duration + eps, out=np.zeros_like(slack), where=duration != 0)
    slack_norm = np.nan_to_num(slack_norm, nan=0.0)
    # Sigmoid sensitivity: steep drop near zero slack
    slack_sensitivity = 0.5 * (np.tanh(-slack_norm / 0.2) + 1.0)
    critical_slack_sensitivity = upward_rank * slack_sensitivity
    
    # Risk-adjusted energy density with quadratic uncertainty scaling
    uncertainty_alpha = 0.7
    scaled_uncertainty = np.power(np.maximum(uncertainty, eps), uncertainty_alpha)
    # Uncertainty-weighted duration amplification
    effective_duration = duration * (1.0 + scaled_uncertainty * (1.0 + uncertainty) + eps)
    risk_energy_density = np.divide(min_incremental_energy, effective_duration, 
                                   out=np.zeros_like(min_incremental_energy), 
                                   where=effective_duration != 0)
    risk_energy_density = np.nan_to_num(risk_energy_density, nan=0.0, posinf=0.0, neginf=0.0)
    norm_energy_density = robust_mad_norm(risk_energy_density)
    
    # Energy score: inverted only in safe region to promote efficiency without risk
    energy_score = is_safe * (-0.35 * norm_energy_density)
    
    # Work consolidation: prefer stable VMs for light sub-DAGs, tolerate risk for heavy ones
    norm_remaining_work = robust_mad_norm(remaining_work)
    unc_work_density = uncertainty / (1.0 + np.clip(norm_remaining_work, 0.0, 10.0) + eps)
    unc_work_density = np.nan_to_num(unc_work_density, nan=0.0, posinf=0.0, neginf=0.0)
    norm_unc_work_density = robust_mad_norm(unc_work_density)
    work_consolidation_bonus = is_safe * (-0.08 * norm_unc_work_density)
    
    # Starvation awareness: relative age normalized by critical-path length
    cp_length_estimate = duration * upward_rank + eps
    relative_age = np.divide(ready_wait_time, cp_length_estimate, 
                            out=np.zeros_like(ready_wait_time), 
                            where=cp_length_estimate != 0)
    relative_age = np.nan_to_num(relative_age, nan=0.0, posinf=0.0, neginf=0.0)
    relative_age = np.clip(relative_age, 0.0, 10.0)
    norm_relative_age = robust_mad_norm(relative_age)
    wait_pressure = (1.0 - is_urgent) * 0.12 * norm_relative_age
    
    # Critical path importance (normalized upward rank)
    total_cp = np.max(upward_rank) if N > 0 else 1.0
    cp_importance = np.divide(upward_rank, total_cp + eps, 
                             out=np.zeros_like(upward_rank), 
                             where=total_cp + eps != 0)
    norm_cp_importance = robust_mad_norm(cp_importance)
    
    # Composite score construction
    # Urgency dominates: hard penalty for lateness, plus critical-latency coupling
    score = np.full(N, 0.0, dtype=np.float64)
    # Assign extreme priority for urgent tasks
    score = np.where(is_urgent, -1000000000000.0, score)
    # For non-urgent: add criticality-weighted slack sensitivity
    score = np.where(is_urgent, score, score + 0.45 * norm_cp_importance * critical_slack_sensitivity)
    # Add energy efficiency incentive only when safe
    score = np.where(is_urgent, score, score + energy_score)
    # Add work consolidation bonus only when safe
    score = np.where(is_urgent, score, score + work_consolidation_bonus)
    # Add starvation relief
    score = np.where(is_urgent, score, score + wait_pressure)
    # Apply unclipped linear urgency for deficit (enhances lateness dominance)
    score = np.where(is_urgent, score, score + 0.05 * norm_deficit)
    
    # Final clipping and sanitization
    score = np.clip(score, -1000000000000.0, 1000000000000.0)
    score = np.nan_to_num(score, nan=1000000000000.0, posinf=1000000000000.0, neginf=-1000000000000.0)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
