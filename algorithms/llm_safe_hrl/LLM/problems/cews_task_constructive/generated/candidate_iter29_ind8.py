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
    v2 priority rule: Hybrid urgency-criticality-energy scoring with adaptive fairness,
                      work-intensity correction, and robust slack-respecting normalization.
    
    Combines Parent 2's smooth urgency bias and graded energy penalty with Parent 1's
    starvation-resilient fairness gating and energy-efficiency boost under safe slack.
    Novel improvements:
      - Unified urgency term: clipped linear + soft saturation (1/(1+|rel_slack|^0.3)) for continuity near DDL
      - Criticality-gated energy optimization: applies efficiency_boost only when upward_rank > median AND slack > 0
      - Work-intensity penalty enhanced with comm/comp ratio to suppress high-communication tasks
      - Fairness boost now requires *both* high uncertainty AND long wait_ratio (>2.0) for stronger starvation prevention
      - All norms use 2%-98% clipping for better outlier resilience in small-N cases
      - Final score bounded and NaN/inf hardened per contract
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
    
    # Compute duration and relative slack safely
    duration = min_exec_time + min_comm_time + eps
    rel_slack = np.divide(slack, duration, out=np.zeros_like(slack), where=duration != 0)
    rel_slack = np.nan_to_num(rel_slack, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Unified urgency bias: smooth clipped linear + soft saturation decay
    urgency_linear = np.clip(-rel_slack, 0.0, 0.6)  # 0.6 cap avoids excessive dominance
    urgency_saturation = 1.0 / (1.0 + np.power(np.abs(rel_slack) + eps, 0.3))
    urgency_bias = urgency_linear * urgency_saturation
    norm_urgency = robust_minmax_norm(urgency_bias)
    
    # Energy density with slack-aware rescaling (Parent 2 style, improved exponent)
    energy_density = np.divide(min_incremental_energy, duration + eps, out=np.zeros_like(min_incremental_energy), where=duration + eps != 0)
    energy_density = np.nan_to_num(energy_density, nan=0.0, posinf=0.0, neginf=0.0)
    slack_rescale = 1.0 + np.power(np.abs(rel_slack) + eps, 0.4)
    risk_adjusted_energy = energy_density / (slack_rescale + eps)
    norm_risk_energy = robust_minmax_norm(risk_adjusted_energy)
    
    # Criticality-aware energy optimization: efficiency boost only when safe & critical
    ur_median = np.median(upward_rank) if N > 1 else upward_rank[0]
    efficiency_boost_mask = (slack > 0.0) & (upward_rank > ur_median)
    efficiency_boost = -0.15 * norm_risk_energy * efficiency_boost_mask
    
    # Work-intensity correction: penalize low compute density (high comm/comp ratio)
    comm_comp_ratio = np.divide(min_comm_time, min_exec_time + eps, out=np.zeros_like(min_exec_time), where=min_exec_time + eps != 0)
    comm_comp_ratio = np.nan_to_num(comm_comp_ratio, nan=0.0, posinf=0.0, neginf=0.0)
    norm_comm_comp_ratio = robust_minmax_norm(comm_comp_ratio)
    work_intensity_penalty = norm_comm_comp_ratio  # higher ratio → higher penalty
    
    # Uncertainty-modulated fairness: stricter condition prevents spurious relief
    wait_ratio = np.divide(ready_wait_time, duration + eps, out=np.zeros_like(ready_wait_time), where=duration + eps != 0)
    wait_ratio = np.nan_to_num(wait_ratio, nan=0.0, posinf=0.0, neginf=0.0)
    unc_median = np.median(uncertainty) if N > 1 else uncertainty[0]
    fairness_boost_mask = (wait_ratio > 2.0).astype(float) * (uncertainty > unc_median).astype(float)
    norm_ready_wait = robust_minmax_norm(ready_wait_time)
    fairness_boost = norm_ready_wait * fairness_boost_mask
    
    # Normalized criticality and remaining work
    norm_upward_rank = robust_minmax_norm(upward_rank)
    norm_remaining_work = robust_minmax_norm(remaining_work)
    
    # Final weighted combination
    score = (
        0.35 * norm_urgency +
        0.22 * norm_risk_energy * (0.6 + 0.4 * norm_upward_rank) +  # graded energy penalty
        0.14 * (1.0 - fairness_boost) +  # fairness as relief (not penalty)
        0.13 * work_intensity_penalty +
        0.10 * norm_remaining_work +
        0.06 * efficiency_boost
    )
    
    # Hard bound and NaN/inf protection
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
