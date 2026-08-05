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
    v2 priority rule: Hard urgency dominance + risk-normalized criticality + 
                      uncertainty-aware energy leverage + starvation-robust fairness + 
                      synergy-aware communication penalty.
    
    Key improvements:
    - Combines Parent 2's strict urgency gating and robust trimmed normalization with 
      Parent 1's feasible-subset energy normalization and starvation-aware wait-per-work.
    - Introduces *critical-path-aware energy leverage*: multiplies risk-normalized energy 
      by upward_rank only when slack <= 0, amplifying energy penalties on high-criticality tasks.
    - Replaces linear wait-time scaling with *feasible-subset trimmed wait-per-work*, 
      normalized robustly and activated only for non-urgent tasks to avoid starvation.
    - Adds *communication-energy synergy term* for urgent+uncertain tasks: product of 
      normalized comm_ratio and energy_score, weighted to jointly penalize high-comm/high-energy.
    - Uses unified robust normalization: trimmed-mean ± 3*MAD (more stable than quantiles).
    - All operations guarded against division-by-zero, NaN, and inf; deterministic and finite.
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

    def robust_trimmed_norm(x):
        if x.size == 0:
            return np.zeros_like(x)
        x_med = np.median(x)
        abs_dev = np.abs(x - x_med)
        mad = np.median(abs_dev) if abs_dev.size > 0 else 0.0
        scale = 3.0 * mad + eps
        if scale < eps:
            return np.zeros_like(x)
        z = (x - x_med) / scale
        return np.clip(z, -3.0, 3.0)

    # Hard urgency flag: tasks violating or at deadline have highest priority
    is_urgent = (slack <= 0.0).astype(np.float64)
    
    # Duration and feasibility mask
    duration = min_exec_time + min_comm_time + eps
    feasible_mask = slack >= -eps
    
    # Critical latency: only penalize long-duration tasks with tight slack
    duration_med = np.median(duration) + eps
    tight_slack_mask = (slack <= 0.25 * duration_med).astype(np.float64)
    long_duration_mask = (duration >= duration_med).astype(np.float64)
    critical_gate = tight_slack_mask * long_duration_mask
    
    # Risk-normalized critical latency: upward_rank scaled by duration and uncertainty
    effective_duration = duration * (1.0 + uncertainty + eps)
    critical_latency_raw = effective_duration * upward_rank
    norm_critical_latency = robust_trimmed_norm(critical_latency_raw)
    
    # Risk-normalized energy: suppress high-energy on uncertain/long VMs
    risk_norm_energy = np.divide(min_incremental_energy, effective_duration, 
                                 out=np.zeros_like(min_incremental_energy), 
                                 where=effective_duration != 0)
    risk_norm_energy = np.where(np.isfinite(risk_norm_energy), risk_norm_energy, 0.0)
    
    # Critical-path-aware energy leverage: amplify energy penalty for urgent high-rank tasks
    energy_leverage = np.where(is_urgent, upward_rank / (np.median(upward_rank) + eps), 1.0)
    leveraged_energy = risk_norm_energy * energy_leverage
    norm_energy = robust_trimmed_norm(leveraged_energy)
    
    # Feasible-subset energy normalization (from Parent 1) for robustness
    if np.any(feasible_mask):
        feasible_energy = risk_norm_energy[feasible_mask]
        q1_f, q3_f = np.quantile(feasible_energy, [0.25, 0.75], method='midpoint')
        iqr_f = q3_f - q1_f + eps
        center_f = np.median(feasible_energy)
        energy_norm_base = (risk_norm_energy - center_f) / iqr_f
        energy_score = -np.clip(energy_norm_base, -4.0, 4.0)
    else:
        energy_score = np.zeros(N)
    
    # Starvation-robust fairness: wait-per-work normalized over feasible subset only
    wait_per_work = ready_wait_time / (remaining_work + eps)
    starvation_mask = feasible_mask & (remaining_work > eps)
    if np.any(starvation_mask):
        feasible_wpw = wait_per_work[starvation_mask]
        wpw_med = np.median(feasible_wpw)
        wpw_abs_dev = np.abs(feasible_wpw - wpw_med)
        wpw_mad = np.median(wpw_abs_dev) if wpw_abs_dev.size > 0 else 0.0
        wpw_scale = 3.0 * wpw_mad + eps
        if wpw_scale < eps:
            wpw_normed = np.zeros_like(wait_per_work)
        else:
            wpw_normed = (wait_per_work - wpw_med) / wpw_scale
            wpw_normed = np.clip(wpw_normed, 0.0, 1.0)
        starvation_penalty = np.where(starvation_mask, wpw_normed * 0.4, 0.0)
    else:
        starvation_penalty = np.zeros(N)
    
    # Communication pressure: ratio of comm time to work, amplified under urgency+uncertainty
    comm_to_work_ratio = min_comm_time / (remaining_work + eps)
    median_unc = np.median(uncertainty) + eps
    urgent_high_uncert_mask = (slack <= 0.0) & (uncertainty > median_unc)
    comm_pressure = np.where(urgent_high_uncert_mask, comm_to_work_ratio * 4.0, comm_to_work_ratio * 0.1)
    norm_comm = robust_trimmed_norm(comm_pressure)
    
    # Communication-energy synergy for urgent+uncertain tasks
    synergy_term = np.where(urgent_high_uncert_mask, energy_score * norm_comm * 0.3, 0.0)
    
    # Normalized remaining work and uncertainty
    norm_remaining_work = robust_trimmed_norm(remaining_work)
    norm_uncertainty = robust_trimmed_norm(uncertainty)
    
    # Final score composition: urgency dominates, then latency, energy, fairness, risk
    score = np.full(N, 0.0, dtype=np.float64)
    
    # Urgent tasks get massive priority boost
    score = np.where(is_urgent, -1e12, score)
    
    # Non-urgent: add latency penalty (gated), energy penalty (gated), fairness, risk terms
    score = np.where(is_urgent, score, score + 0.35 * norm_critical_latency)
    score = np.where(is_urgent, score, score + 0.25 * norm_energy * critical_gate)
    score = np.where(is_urgent, score, score + 0.2 * starvation_penalty)
    score = np.where(is_urgent, score, score + 0.12 * norm_comm)
    score = np.where(is_urgent, score, score + 0.08 * norm_remaining_work)
    score = np.where(is_urgent, score, score + 0.05 * norm_uncertainty)
    
    # Add synergy term unconditionally but only active where mask applies
    score += synergy_term
    
    # Clamp and sanitize
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
