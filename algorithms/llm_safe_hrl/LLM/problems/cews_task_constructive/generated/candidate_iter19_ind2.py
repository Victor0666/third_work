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
    v2 priority rule: Hard urgency dominance + tanh-slack pressure + MAD-normalized multi-scale gating +
                     risk-weighted starvation relief + uncertainty-coupled criticality amplification.
    
    Key mutations vs v1:
    - Replaces piecewise rel_slack thresholding with smooth, quantile-scaled tanh urgency signal.
    - Uses robust MAD (median absolute deviation) instead of percentile clipping for normalization → more stable on small N.
    - Introduces 'risk-adjusted starvation relief': wait_time prioritization *only* when uncertainty is high AND slack is positive.
    - Replaces static energy_penalty_mask with dynamic coupling: energy_density weighted by tanh(slack/median_duration).
    - Adds criticality amplification: upward_rank scaled by tanh(uncertainty * slack_sensitivity), boosting uncertain critical tasks.
    - Removes fixed work_threshold gating; uses relative waiting (wait_time / median_duration) for fairness.
    - All norms use epsilon-guarded MAD scaling with fallback to zeros (not min-max) for numerical stability.
    """
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=float).copy()
    min_comm_time = np.asarray(min_comm_time, dtype=float).copy()
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float).copy()
    slack = np.asarray(slack, dtype=float).copy()
    upward_rank = np.asarray(upward_rank, dtype=float).copy()
    remaining_work = np.asarray(remaining_work, dtype=float).copy()
    ready_wait_time = np.asarray(ready_wait_time, dtype=float).copy()
    uncertainty = np.asarray(uncertainty, dtype=float).copy()
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)

    # Robust MAD-based normalization: more stable than percentile clipping for small N
    def robust_mad_norm(x):
        if x.size == 0:
            return np.zeros_like(x)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        if mad < eps:
            return np.zeros_like(x)
        return (x - med) / (mad + eps)

    # Hard urgency dominance: zero-tolerance deadline enforcement
    is_urgent = (slack <= 0.0).astype(float)
    
    # Duration base (execution + communication), guard against zero
    duration = min_exec_time + min_comm_time + eps
    
    # Tanh-scaled urgency: smooth, bounded, scale-invariant pressure from slack
    # Use median duration to normalize slack magnitude → makes tanh input unitless and adaptive
    median_dur = np.median(duration) + eps
    tanh_slack_input = np.clip(slack / median_dur, -10.0, 10.0)
    tanh_urgency = 0.5 * (1.0 - np.tanh(tanh_slack_input))  # [0,1]: 1=most urgent
    
    # Critical latency: duration weighted by upward_rank, amplified by uncertainty only if slack > 0
    # Avoids over-amplifying already-urgent tasks (which get hard priority anyway)
    unc_coupled_rank = upward_rank * (1.0 + np.where(slack > 0, uncertainty, 0.0))
    critical_latency_raw = duration * (1.0 + 0.8 * robust_mad_norm(unc_coupled_rank))
    norm_critical_latency = robust_mad_norm(critical_latency_raw)
    
    # Energy density: incremental energy per unit time, now gated by tanh_urgency (not binary mask)
    # High urgency → lower energy penalty weight; low urgency → full energy consideration
    energy_density = np.divide(min_incremental_energy, duration, out=np.zeros_like(min_incremental_energy), where=duration != 0)
    energy_density = np.nan_to_num(energy_density, nan=0.0, posinf=0.0, neginf=0.0)
    norm_energy_density = robust_mad_norm(energy_density)
    # Energy weight decays smoothly with urgency: 1.0 at slack→-∞, ~0.2 at slack=median_dur
    energy_weight = 0.2 + 0.8 * (1.0 - tanh_urgency)
    energy_term = energy_weight * norm_energy_density
    
    # Starvation relief: only activated under *positive slack AND high uncertainty*
    # Prevents long waits from accumulating when system is uncertain but not yet urgent
    high_uncertainty = (uncertainty > np.percentile(uncertainty, 80) + eps).astype(float)
    safe_slack = (slack > 0.0).astype(float)
    starvation_gate = high_uncertainty * safe_slack
    # Wait-time normalized by duration (not work) → fairness per latency budget, not MI
    wait_per_duration = np.divide(ready_wait_time, duration, out=np.zeros_like(ready_wait_time), where=duration != 0)
    wait_per_duration = np.nan_to_num(wait_per_duration, nan=0.0, posinf=0.0, neginf=0.0)
    norm_wait_rel = robust_mad_norm(wait_per_duration)
    starvation_relief = starvation_gate * (0.3 * norm_wait_rel)  # positive score = lower priority → invert later
    
    # Remaining work term: discourage scheduling tiny leaf tasks when large upstream work remains
    # Inverted: higher work → higher priority (lower score), so subtract from base
    norm_remaining_work = robust_mad_norm(remaining_work)
    work_bonus = -0.15 * norm_remaining_work  # negative bonus → improves priority
    
    # Base score: start neutral, then apply all terms
    base_score = np.full(N, 0.0, dtype=float)
    
    # Assemble final score: smallest = highest priority
    # Urgent tasks get guaranteed min score (not offset)
    score = np.where(is_urgent, -1e12, base_score)
    # Add latency, energy, starvation, work terms — all scaled and normalized
    score = np.where(is_urgent, score, 
                     score 
                     + 0.32 * norm_critical_latency 
                     + 0.26 * energy_term 
                     + 0.14 * starvation_relief 
                     + 0.13 * (-tanh_urgency)  # direct urgency boost: lower = better
                     + work_bonus)
    
    # Clip extreme values and sanitize NaN/inf
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
