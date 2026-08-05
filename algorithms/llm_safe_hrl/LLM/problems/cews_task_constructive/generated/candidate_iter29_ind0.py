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
    v2 priority rule: Hybrid adaptive urgency + criticality-gated risk-energy tradeoff +
                      uncertainty-thresholded fairness + work-density regularization +
                      robust slack-aware normalization.
    
    Key synthesis improvements:
    - Combines Parent 2's smooth urgency bias (clipped linear -rel_slack) with Parent 1's
      explicit lateness penalty for hard DDL enforcement when slack <= 0, but avoids overflow
      via bounded scaling: penalty = -1e10 * (1 + 0.05 * |slack|) for slack <= 0.
    - Uses graded energy penalty weighted by upward_rank (Parent 2), but gates it with slack > 0
      to prevent energy optimization at deadline violation cost (Parent 1 discipline).
    - Uncertainty-modulated fairness (Parent 2) enhanced with dual threshold: wait_ratio > 1.5
      AND uncertainty > median, preserving Parent 1's robustness against spurious boosts.
    - Adds work-intensity correction (Parent 2) and extends it with communication-to-compute ratio
      penalty to reduce energy-wasteful high-comm tasks.
    - Introduces novel slack-proximity gating on energy density: applies stronger energy preference
      only when rel_slack ∈ [0.1, 0.8], avoiding over-prioritization of ultra-tight or relaxed tasks.
    - All normalization uses 2%-98% clipping for better outlier suppression; all divisions guarded;
      all NaN/inf replaced deterministically; fully deterministic.
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
    
    def robust_minmax_norm(x):
        if x.size == 0:
            return np.zeros_like(x)
        if N == 1:
            return np.zeros_like(x)
        p02 = np.percentile(x, 2.0, method='lower')
        p98 = np.percentile(x, 98.0, method='higher')
        x_clipped = np.clip(x, p02, p98)
        x_min = np.min(x_clipped)
        x_max = np.max(x_clipped)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x_clipped - x_min) / (x_max - x_min + eps)
    
    # Compute base duration and relative slack
    duration = min_exec_time + min_comm_time + eps
    rel_slack = np.divide(slack, duration, out=np.zeros_like(slack), where=duration != 0)
    rel_slack = np.nan_to_num(rel_slack, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Hard deadline enforcement: bounded lateness penalty
    lateness_mask = slack <= 0.0
    abs_slack = np.abs(slack)
    lateness_penalty = np.where(lateness_mask, -1e10 * (1.0 + 0.05 * abs_slack), 0.0)
    
    # Smooth urgency bias for positive slack region (Parent 2), clipped to [0, 0.45]
    urgency_bias_raw = np.clip(-rel_slack, 0.0, 0.45)
    norm_urgency_bias = robust_minmax_norm(urgency_bias_raw)
    
    # Energy density with slack-aware rescaling: 1/(1 + sqrt(|rel_slack|+eps)) → soft saturation
    energy_density = np.divide(min_incremental_energy, duration + eps, 
                              out=np.zeros_like(min_incremental_energy), 
                              where=duration + eps != 0)
    energy_density = np.nan_to_num(energy_density, nan=0.0, posinf=0.0, neginf=0.0)
    slack_rescale = 1.0 + np.sqrt(np.abs(rel_slack) + eps)
    risk_adjusted_energy = energy_density / (slack_rescale + eps)
    norm_risk_energy = robust_minmax_norm(risk_adjusted_energy)
    
    # Criticality gating for energy penalty: only apply when slack > 0 (hard constraint safety)
    energy_penalty_weight = np.where(slack > 0.0, 0.5 + 0.5 * robust_minmax_norm(upward_rank), 0.0)
    
    # Work intensity: penalize low compute density (high comm/exec ratio) and low total work density
    comm_to_exec_ratio = np.divide(min_comm_time, min_exec_time + eps, 
                                  out=np.zeros_like(min_comm_time), 
                                  where=min_exec_time + eps != 0)
    comm_to_exec_ratio = np.nan_to_num(comm_to_exec_ratio, nan=0.0, posinf=0.0, neginf=0.0)
    work_density = np.divide(remaining_work, duration + eps, 
                            out=np.zeros_like(remaining_work), 
                            where=duration + eps != 0)
    work_density = np.nan_to_num(work_density, nan=0.0, posinf=0.0, neginf=0.0)
    norm_work_density = robust_minmax_norm(work_density)
    work_intensity_penalty = 0.6 * (1.0 - norm_work_density) + 0.4 * comm_to_exec_ratio
    
    # Uncertainty-modulated fairness: gated by both high wait_ratio AND high uncertainty
    wait_ratio = np.divide(ready_wait_time, duration + eps, 
                          out=np.zeros_like(ready_wait_time), 
                          where=duration + eps != 0)
    wait_ratio = np.nan_to_num(wait_ratio, nan=0.0, posinf=0.0, neginf=0.0)
    unc_median = np.median(uncertainty) if N > 1 else uncertainty[0]
    fairness_boost_mask = (wait_ratio > 1.5) & (uncertainty > unc_median)
    norm_ready_wait = robust_minmax_norm(ready_wait_time)
    fairness_boost = norm_ready_wait * fairness_boost_mask.astype(np.float64)
    
    # Slack-proximity gating for energy preference: strongest effect when 0.1 <= rel_slack <= 0.8
    slack_gate = np.clip(1.0 - 2.0 * np.abs(rel_slack - 0.45), 0.0, 1.0)
    slack_gate = np.where((rel_slack >= 0.1) & (rel_slack <= 0.8), slack_gate, 0.0)
    
    # Remaining work importance (normalized)
    norm_remaining_work = robust_minmax_norm(remaining_work)
    
    # Final score composition
    score = (
        0.35 * norm_urgency_bias +                           # primary deadline adherence signal
        0.28 * norm_risk_energy * energy_penalty_weight * slack_gate +  # energy efficiency under safe slack
        0.14 * work_intensity_penalty +                      # communication & density regularization
        0.12 * (1.0 - fairness_boost) +                      # fairness boost reduces priority (since smaller = better)
        0.08 * norm_remaining_work +                         # critical path support
        0.03 * robust_minmax_norm(upward_rank)               # residual criticality reinforcement
    )
    
    # Apply hard lateness penalty last
    score = lateness_penalty + score
    
    # Final sanitization
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
