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
    v2 priority rule: Hard deadline dominance + risk-gated energy-aware criticality + 
    MAD-robust starvation relief + uncertainty-normalized slack urgency.
    
    Key mutations vs v1:
    - Replaces percentile-based clipping with median ± 3*MAD for outlier-resilient normalization
    - Uses *inverse slack* (not relative slack) for urgency, clipped to [0.1, 100] and scaled by uncertainty-aware confidence
    - Introduces "criticality-energy tension" term: only penalizes energy when upward_rank > median AND slack < median_slack
    - Starvation relief now uses wait_time / (remaining_work + eps) normalized via robust z-score (median/MAD), not min-max
    - Adds "workload density" gate: activates wait pressure only when remaining_work is above median (avoids biasing tiny tasks)
    - Uncertainty coupling applied *before* normalization to preserve risk-grading fidelity
    - All divisions guarded; NaN/inf replaced deterministically; no side effects.
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

    def robust_mad_norm(x):
        if x.size == 0:
            return np.zeros_like(x)
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        # Avoid degenerate case: if MAD ≈ 0, use std as fallback (still bounded)
        scale = mad if mad > eps else np.std(x) + eps
        z_score = (x - med) / (scale + eps)
        # Clip extreme outliers to [-3, 3] → ensures stable ranking across sparse/dense sets
        return np.clip(z_score, -3.0, 3.0)

    def robust_z_clip(x):
        z = robust_mad_norm(x)
        return np.clip(z, -3.0, 3.0)

    # Hard urgency override: negative or zero slack → guaranteed highest priority
    is_urgent = (slack <= 0.0).astype(float)
    
    # Duration for scaling (execution + communication)
    duration = min_exec_time + min_comm_time + eps
    
    # Inverse urgency: 1/(|slack|+eps), clipped and scaled by uncertainty confidence
    # Higher uncertainty reduces urgency weight (less trust in slack estimate)
    inv_slack_raw = np.divide(1.0, np.abs(slack) + eps, out=np.zeros_like(slack), where=(np.abs(slack) + eps) != 0)
    inv_slack_confidence = 1.0 / (1.0 + uncertainty + eps)  # confidence ∈ (0,1]
    inv_slack = inv_slack_raw * inv_slack_confidence
    inv_slack = np.clip(inv_slack, 0.1, 100.0)  # prevent explosion, retain scale-invariance
    
    # Critical path importance: upward_rank normalized robustly
    norm_upward_rank = robust_z_clip(upward_rank)
    
    # Energy density: marginal energy per unit duration, uncertainty-weighted
    energy_density = np.divide(min_incremental_energy, duration, out=np.zeros_like(min_incremental_energy), where=duration != 0)
    # Apply uncertainty scaling: higher uncertainty → lower effective penalty weight
    energy_density_adj = energy_density / (1.0 + uncertainty + eps)
    norm_energy_density = robust_z_clip(energy_density_adj)
    
    # Criticality-energy tension gate: only activate energy penalty when both conditions hold
    median_upward = np.median(upward_rank) + eps
    median_slack = np.median(slack) + eps
    tension_mask = ((upward_rank > median_upward) & (slack < median_slack)).astype(float)
    
    # Starvation relief: wait_time per unit work, normalized robustly
    wait_per_work = np.divide(ready_wait_time, remaining_work + eps, 
                             out=np.zeros_like(ready_wait_time), where=(remaining_work + eps) != 0)
    norm_wait_per_work = robust_z_clip(wait_per_work)
    
    # Workload density gate: only apply wait pressure for medium/large tasks
    median_work = np.median(remaining_work) + eps
    work_density_gate = (remaining_work >= median_work).astype(float)
    
    # Base score components (all z-scores, so centered at 0)
    base_score = np.full(N, 0.0, dtype=float)
    
    # Composite score: lower = better
    # Urgent tasks get hard override
    score = np.where(is_urgent, -1e12, base_score)
    
    # Non-urgent tasks: weighted sum of normalized terms
    # Critical latency proxy: duration × (1 + norm_upward_rank) → favors fast execution on critical paths
    crit_latency_proxy = duration * (1.0 + 0.5 * norm_upward_rank)
    norm_crit_latency = robust_z_clip(crit_latency_proxy)
    
    # Energy penalty only under tension
    energy_penalty = norm_energy_density * tension_mask
    
    # Starvation relief gated by workload density
    wait_relief = norm_wait_per_work * work_density_gate
    
    # Uncertainty-modulated urgency (already computed as inv_slack, now normalized)
    norm_inv_slack = robust_z_clip(inv_slack)
    
    # Combine with empirically tuned weights (preserving total ~1.0 weight budget)
    score = np.where(
        is_urgent,
        score,
        score 
        + 0.35 * norm_crit_latency 
        + 0.25 * energy_penalty 
        + 0.20 * (-wait_relief)  # negative: longer wait → lower score → higher priority
        + 0.15 * (-norm_inv_slack)  # negative: higher urgency → lower score
        + 0.05 * robust_z_clip(uncertainty)  # slight penalty for high uncertainty (risk aversion)
    )
    
    # Final bounds and NaN cleanup
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
