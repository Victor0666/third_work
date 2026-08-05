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
    v2 priority rule: Deadline-feasibility-dominant + criticality-gated risk-energy efficiency + 
                      uncertainty-weighted starvation relief + robust percentile normalization.
    
    Key improvements:
    - Hard feasibility dominance: violated tasks get score = -inf (clipped to large negative) 
      for strict deadline adherence.
    - Criticality-gated energy efficiency: only penalizes energy on high-upward-rank paths 
      under tight slack (rel_slack <= 0.3), avoiding wasteful optimization on low-impact branches.
    - Uncertainty-weighted starvation relief: boosts long-waiting tasks *only* when both 
      uncertainty is high AND upward_rank is above adaptive median+IQR threshold.
    - Robust 5%-95% percentile normalization (stable for small N, avoids z-score fragility).
    - Slack-rescaled energy density: energy_density / (1 + |rel_slack|) — stronger penalty when slack shrinks.
    - No additive sign inversion: all terms are positive and monotonically increasing in score 
      (lower score = higher priority), preserving ranking stability.
    - Strict numerical hygiene: eps guards, nan_to_num with explicit fill values, finite clipping.
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
    
    # Normalize any input that might be zero or degenerate
    def robust_minmax_norm(x):
        if x.size == 0:
            return np.zeros_like(x)
        p05 = np.percentile(x, 5.0)
        p95 = np.percentile(x, 95.0)
        x_clipped = np.clip(x, p05, p95)
        x_min = np.min(x_clipped)
        x_max = np.max(x_clipped)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x_clipped - x_min) / (x_max - x_min + eps)
    
    # Compute base duration and relative slack
    duration = min_exec_time + min_comm_time + eps
    rel_slack = np.divide(slack, duration, out=np.zeros_like(slack), where=duration != 0)
    rel_slack = np.nan_to_num(rel_slack, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Energy density normalized by duration; rescaled by slack tightness
    energy_density = np.divide(min_incremental_energy, duration + eps, 
                               out=np.zeros_like(min_incremental_energy), 
                               where=duration + eps != 0)
    slack_rescale = 1.0 + np.abs(rel_slack)
    risk_adjusted_energy = energy_density / (slack_rescale + eps)
    
    # Criticality gating: identify high-criticality tasks using robust adaptive threshold
    q50_ur = np.median(upward_rank)
    q25_ur, q75_ur = np.percentile(upward_rank, [25.0, 75.0])
    iqr_ur = q75_ur - q25_ur + eps
    crit_threshold = q50_ur + 1.0 * iqr_ur  # less aggressive than Parent 1's 1.5*IQR
    high_crit_mask = (upward_rank > crit_threshold).astype(float)
    
    # Tight-slack mask: rel_slack <= 0.3 → imminent deadline pressure
    tight_slack_mask = (rel_slack <= 0.3).astype(float)
    
    # Energy penalty applied only where both criticality and urgency coincide
    energy_penalty_mask = high_crit_mask * tight_slack_mask
    
    # Normalize components independently
    norm_risk_energy = robust_minmax_norm(risk_adjusted_energy)
    norm_upward_rank = robust_minmax_norm(upward_rank)
    norm_ready_wait = robust_minmax_norm(ready_wait_time)
    norm_uncertainty = robust_minmax_norm(uncertainty)
    
    # Starvation relief: only for high-uncertainty + high-criticality + long wait
    wait_ratio = np.divide(ready_wait_time, duration + eps, 
                           out=np.zeros_like(ready_wait_time), 
                           where=duration + eps != 0)
    wait_ratio = np.nan_to_num(wait_ratio, nan=0.0, posinf=0.0, neginf=0.0)
    starvation_boost = norm_ready_wait * norm_uncertainty * high_crit_mask * (wait_ratio > 1.2).astype(float)
    
    # Base score starts neutral; feasibility dominates
    base_score = np.full(N, 1.0, dtype=float)
    
    # Feasibility boost: violated tasks get massive priority (lowest score)
    is_violated = (slack < 0.0).astype(float)
    score = np.where(is_violated, -1000000000000.0, base_score)
    
    # Add weighted contributions — all positive weights, monotonic in score
    # Weights sum to 1.0: urgency proxy (rel_slack) + energy + starvation + criticality
    score = np.where(is_violated, score,
                     score + 
                     0.40 * robust_minmax_norm(np.clip(-rel_slack, 0.0, 10.0)) +  # urgency: higher -rel_slack → lower score
                     0.28 * norm_risk_energy * energy_penalty_mask +              # gated energy efficiency
                     0.18 * starvation_boost +                                   # uncertainty-aware starvation relief
                     0.14 * (1.0 - norm_upward_rank))                            # prefer high-criticality (lower score for high ur)
    
    # Final numerical cleanup
    score = np.clip(score, -1000000000000.0, 1000000000000.0)
    score = np.nan_to_num(score, nan=1000000000000.0, posinf=1000000000000.0, neginf=-1000000000000.0)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
