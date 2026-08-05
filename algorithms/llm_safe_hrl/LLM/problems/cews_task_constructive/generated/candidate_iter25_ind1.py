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
    v2 mutation: Tighter urgency gating + slack-aware energy normalization + 
    starvation-robust wait-time scaling + uncertainty-coupled criticality decay.
    
    Key mutations:
    - Urgency threshold tightened to slack <= 0.1 (not just <= 0) for smoother 
      transition near deadline, while still guaranteeing absolute priority for slack <= 0.
    - Energy penalty now normalized *per-slack-band*: low-slack tasks use local 5%-95% 
      range over urgent subset only, preserving discriminative power under volatility.
    - Wait-time fairness uses adaptive percentile gating: activates only when 
      ready_wait_time > median(ready_wait_time) AND remaining_work < 25th percentile, 
      reducing false starvation signals in bursty workloads.
    - Criticality decay: upward_rank scaled by exp(-0.5 * max(0, rel_slack)), softening 
      its influence as slack increases — avoids over-prioritizing non-urgent critical paths.
    - Uncertainty coupling now multiplicative with *normalized slack pressure*, not additive,
      ensuring risk amplification only where it matters (tight slack + high rank).
    - All robust normalizations now include explicit zero-range fallback to uniform 0.5.
    """
    eps = 1e-8
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

    def robust_minmax_norm(x):
        if x.size == 0:
            return np.zeros_like(x)
        p01 = np.percentile(x, 1.0)
        p99 = np.percentile(x, 99.0)
        x_clipped = np.clip(x, p01, p99)
        x_min = np.min(x_clipped)
        x_max = np.max(x_clipped)
        if x_max - x_min < eps:
            return np.full_like(x, 0.5)  # fallback: neutral score
        return (x_clipped - x_min) / (x_max - x_min + eps)

    # Absolute urgency: slack <= 0 gets hard minimum; slack <= 0.1 gets strong boost
    is_critical_urgent = (slack <= 0.0).astype(float)
    is_urgent = (slack <= 0.1).astype(float)
    
    duration = min_exec_time + min_comm_time + eps
    rel_slack = np.divide(slack, duration, out=np.zeros_like(slack), where=duration != 0)
    rel_slack = np.where(np.isfinite(rel_slack), rel_slack, 0.0)
    
    # Criticality decay: attenuate upward_rank for loose slack
    slack_decay = np.exp(-0.5 * np.maximum(0.0, rel_slack))
    decayed_upward_rank = upward_rank * slack_decay
    
    # Critical latency: now uses decayed rank and robust norm
    critical_latency_raw = duration * (1.0 + 0.7 * robust_minmax_norm(decayed_upward_rank))
    norm_critical_latency = robust_minmax_norm(critical_latency_raw)
    
    # Energy density: risk-adjusted per-task marginal energy per duration
    energy_density = np.divide(min_incremental_energy, duration + eps, 
                              out=np.zeros_like(min_incremental_energy), 
                              where=duration + eps != 0)
    
    # Urgent-subset energy normalization: improves discrimination under tight slack
    urgent_mask = (slack <= 0.1)
    if np.any(urgent_mask):
        urgent_energy = energy_density[urgent_mask]
        p05 = np.percentile(urgent_energy, 5.0)
        p95 = np.percentile(urgent_energy, 95.0)
        clipped_urgent = np.clip(urgent_energy, p05, p95)
        urgent_min = np.min(clipped_urgent)
        urgent_max = np.max(clipped_urgent)
        if urgent_max - urgent_min < eps:
            norm_energy_density_urgent = np.full_like(urgent_energy, 0.5)
        else:
            norm_energy_density_urgent = (clipped_urgent - urgent_min) / (urgent_max - urgent_min + eps)
        norm_energy_density = np.zeros_like(energy_density)
        norm_energy_density[urgent_mask] = norm_energy_density_urgent
        # For non-urgent: fall back to global robust norm
        non_urgent_mask = ~urgent_mask
        if np.any(non_urgent_mask):
            norm_energy_density[non_urgent_mask] = robust_minmax_norm(energy_density[non_urgent_mask])
    else:
        norm_energy_density = robust_minmax_norm(energy_density)
    
    # Energy penalty: only applied to top 30% of upward_rank *within urgent set*
    rank_threshold_urgent = np.percentile(upward_rank[urgent_mask], 70.0) + eps if np.any(urgent_mask) else np.percentile(upward_rank, 70.0) + eps
    high_rank_mask = (upward_rank > rank_threshold_urgent).astype(float)
    energy_penalty_mask = is_urgent * high_rank_mask
    energy_penalty = norm_energy_density * energy_penalty_mask
    
    # Adaptive wait-time fairness: activate only if waiting long *and* low remaining work
    wait_median = np.median(ready_wait_time) if N > 0 else 0.0
    work_p25 = np.percentile(remaining_work, 25.0) + eps
    wait_gate = ((ready_wait_time > wait_median) & (remaining_work < work_p25)).astype(float)
    norm_wait_time = robust_minmax_norm(ready_wait_time)
    wait_penalty = (1.0 - is_critical_urgent) * norm_wait_time * wait_gate
    
    # Slack pressure: inverse-linear near deadline, bounded and smoothed
    abs_rel_slack = np.abs(rel_slack) + eps
    slack_pressure = np.clip(1.0 / (abs_rel_slack + 0.1), 0.05, 20.0)  # smoother than pure 1/x
    
    # Uncertainty coupling: multiplicative with slack_pressure and decayed rank
    uncertainty_boost = uncertainty * slack_pressure * robust_minmax_norm(decayed_upward_rank)
    norm_uncertainty_boost = robust_minmax_norm(uncertainty_boost)
    
    # Remaining work bias: slightly favor smaller remaining_work *only* when not urgent
    norm_remaining_work = robust_minmax_norm(remaining_work)
    work_bias = (1.0 - is_urgent) * (1.0 - norm_remaining_work)  # smaller work → higher bias → lower score
    
    # Base composition: weighted sum, with urgent override
    base_score = np.full(N, 1.0, dtype=float)
    score = np.where(is_critical_urgent, -1e12, base_score)
    
    # Non-urgent scoring components
    score = np.where(is_critical_urgent, score,
                     score + 
                     0.25 * norm_critical_latency +
                     0.22 * energy_penalty +
                     0.15 * wait_penalty +
                     0.12 * norm_uncertainty_boost +
                     0.08 * work_bias +
                     0.08 * (1.0 - robust_minmax_norm(slack))  # slack proximity bonus
                    )
    
    # Final clipping and NaN/inf cleanup
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
