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
    v2 priority rule: Hybrid urgency-energy-criticality with adaptive fairness and robust risk coupling.
    
    Key improvements:
    - Absolute urgency dominance (slack <= 0 → -1e12) from Parent 2 for guaranteed deadline adherence.
    - Slack-aware energy normalization per urgency band (Parent 1's urgent/non-urgent stratification).
    - Criticality decay via exp(-0.5 * max(0, rel_slack)) to soften rank over-prioritization when slack is ample.
    - Adaptive starvation guard: wait-per-work ratio gated by dynamic work threshold (5th percentile), not static median.
    - Uncertainty amplification only where critical (tight slack AND high upward_rank), multiplicative and normalized.
    - All normalizations use 1%-99% clipping + min-max with degenerate-range fallback to zero.
    - Strict NaN/inf/zero protection throughout; deterministic output for identical inputs.
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
    
    def robust_minmax_norm(x):
        if x.size == 0:
            return np.zeros_like(x)
        p01 = np.percentile(x, 1.0)
        p99 = np.percentile(x, 99.0)
        x_clipped = np.clip(x, p01, p99)
        x_min = np.min(x_clipped)
        x_max = np.max(x_clipped)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x_clipped - x_min) / (x_max - x_min + eps)
    
    # Absolute urgency enforcement: slack <= 0 always wins
    is_urgent = (slack <= 0.0).astype(float)
    
    # Duration and relative slack for risk-aware scaling
    duration = min_exec_time + min_comm_time + eps
    rel_slack = np.divide(slack, duration, out=np.zeros_like(slack), where=duration != 0)
    rel_slack = np.where(np.isfinite(rel_slack), rel_slack, 0.0)
    
    # Criticality decay: soften upward_rank influence as slack increases
    slack_decay = np.exp(-0.5 * np.maximum(0.0, rel_slack))
    decayed_upward_rank = upward_rank * slack_decay
    
    # Critical latency: duration weighted by decayed rank
    critical_latency_raw = duration * (1.0 + 0.7 * robust_minmax_norm(decayed_upward_rank))
    norm_critical_latency = robust_minmax_norm(critical_latency_raw)
    
    # Energy density: incremental energy per unit duration
    energy_density = np.divide(min_incremental_energy, duration + eps, out=np.zeros_like(min_incremental_energy), where=duration + eps != 0)
    
    # Urgent vs non-urgent energy normalization for discriminative power
    urgent_mask = slack <= 0.1
    if np.any(urgent_mask):
        urgent_energy = energy_density[urgent_mask]
        p05 = np.percentile(urgent_energy, 5.0)
        p95 = np.percentile(urgent_energy, 95.0)
        clipped_urgent = np.clip(urgent_energy, p05, p95)
        urgent_min = np.min(clipped_urgent)
        urgent_max = np.max(clipped_urgent)
        if urgent_max - urgent_min < eps:
            norm_energy_density_urgent = np.zeros_like(urgent_energy)
        else:
            norm_energy_density_urgent = (clipped_urgent - urgent_min) / (urgent_max - urgent_min + eps)
        norm_energy_density = np.zeros_like(energy_density)
        norm_energy_density[urgent_mask] = norm_energy_density_urgent
        non_urgent_mask = ~urgent_mask
        if np.any(non_urgent_mask):
            norm_energy_density[non_urgent_mask] = robust_minmax_norm(energy_density[non_urgent_mask])
    else:
        norm_energy_density = robust_minmax_norm(energy_density)
    
    # Energy penalty: only activate for high-criticality *and* tight-slack tasks
    rank_threshold = np.percentile(upward_rank, 75.0) + eps
    tight_slack_mask = (rel_slack <= 0.3).astype(float)
    high_rank_mask = (upward_rank > rank_threshold).astype(float)
    energy_penalty_mask = tight_slack_mask * high_rank_mask
    energy_penalty = norm_energy_density * energy_penalty_mask
    
    # Adaptive fairness: starvation guard based on wait-per-work ratio
    wait_per_work = np.divide(ready_wait_time, remaining_work + eps, out=np.zeros_like(ready_wait_time), where=remaining_work + eps != 0)
    work_threshold = np.percentile(remaining_work, 5.0) + eps
    wait_gate = (remaining_work >= work_threshold).astype(float)
    norm_wait_time = robust_minmax_norm(ready_wait_time)
    wait_penalty = (1.0 - is_urgent) * norm_wait_time * wait_gate
    
    # Uncertainty coupling: only amplify where both slack pressure and criticality are high
    abs_rel_slack = np.abs(rel_slack) + eps
    slack_sensitivity = np.clip(1.0 / abs_rel_slack, 0.1, 10.0)
    rank_sensitivity = robust_minmax_norm(upward_rank)
    uncertainty_boost = uncertainty * slack_sensitivity * rank_sensitivity
    norm_uncertainty_boost = robust_minmax_norm(uncertainty_boost)
    
    # Work bias: favor smaller remaining work when not urgent (reduces fragmentation)
    norm_remaining_work = robust_minmax_norm(remaining_work)
    work_bias = (1.0 - is_urgent) * (1.0 - norm_remaining_work)
    
    # Base score and composition
    base_score = np.full(N, 1.0, dtype=float)
    score = np.where(is_urgent, -1000000000000.0, base_score)
    score = np.where(
        is_urgent,
        score,
        score + 
        0.26 * norm_critical_latency + 
        0.23 * energy_penalty + 
        0.14 * wait_penalty + 
        0.11 * norm_uncertainty_boost + 
        0.07 * work_bias + 
        0.09 * (1.0 - robust_minmax_norm(slack))
    )
    
    # Final safeguard: clip and replace invalid values deterministically
    score = np.clip(score, -1000000000000.0, 1000000000000.0)
    score = np.nan_to_num(score, nan=1000000000000.0, posinf=1000000000000.0, neginf=-1000000000000.0)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
