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
    v2 priority rule: Hard deadline enforcement + risk-gated energy-aware criticality + 
    MAD-robust starvation relief + uncertainty-normalized slack sensitivity.
    
    Key mutations vs v1:
    - Replaces percentile-based clipping with median ± 3*MAD for outlier-resilient normalization
    - Uses *inverse slack* (not relative slack) with adaptive epsilon scaling to preserve urgency signal at large durations
    - Introduces 'risk-adjusted criticality' = upward_rank * (1 + uncertainty), weighted by slack proximity
    - Energy penalty now gated by *both* tight slack AND high risk-adjusted criticality (not just rank threshold)
    - Starvation relief uses wait_per_work *and* normalized waiting time, scaled by workload density (not fixed percentile gate)
    - All components fused via convex combination with learned weights (0.35/0.3/0.2/0.15) preserving urgency dominance
    - Strict finite-value guarantees: no inf/nan propagation, all divisions guarded, final clip to [-1e12, 1e12]
    """
    eps = 1e-8
    # Convert and copy inputs to avoid mutation
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

    # Robust normalization: median ± 3*MAD (more outlier-resistant than percentile)
    def robust_mad_norm(x):
        if x.size == 0:
            return np.zeros_like(x)
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        if mad < eps:
            return np.zeros_like(x)
        normed = (x - med) / (3.0 * mad + eps)
        # Clamp to [-1, 1] range for stability
        return np.clip(normed, -1.0, 1.0)

    # Hard urgency override: negative or zero slack → guaranteed highest priority
    is_urgent = (slack <= 0.0).astype(float)
    
    # Duration for normalization (execution + communication)
    duration = min_exec_time + min_comm_time + eps
    
    # Inverse slack with adaptive epsilon: preserves urgency signal even for long tasks
    # Avoids division-by-zero and handles large positive slack gracefully
    inv_slack = np.divide(1.0, np.abs(slack) + eps, out=np.zeros_like(slack), where=np.abs(slack) + eps != 0)
    # Scale down large inv_slack to prevent dominance; cap at 10x median
    inv_slack_med = np.median(inv_slack) if N > 0 else 1.0
    inv_slack = np.clip(inv_slack, 0.0, 10.0 * max(inv_slack_med, eps))
    
    # Risk-adjusted criticality: upward rank amplified by uncertainty, then gated by slack proximity
    risk_criticality = upward_rank * (1.0 + uncertainty)
    # Tight-slack mask: slack within 10% of duration or negative
    tight_slack_mask = (slack <= 0.1 * duration).astype(float)
    
    # Energy density: marginal energy per unit duration, risk-normalized
    energy_density = np.divide(min_incremental_energy, duration, out=np.zeros_like(min_incremental_energy), where=duration != 0)
    # Energy penalty only activates for both tight slack AND high risk-criticality
    high_risk_crit_mask = (risk_criticality >= np.percentile(risk_criticality, 70.0) + eps).astype(float)
    energy_penalty_mask = tight_slack_mask * high_risk_crit_mask
    
    # Starvation relief: wait_per_work + normalized wait time, scaled by workload density
    wait_per_work = np.divide(ready_wait_time, remaining_work + eps, out=np.zeros_like(ready_wait_time), where=remaining_work + eps != 0)
    # Workload density proxy: fraction of tasks above median work
    work_density = np.mean(remaining_work >= np.median(remaining_work) + eps) if N > 0 else 0.5
    wait_penalty = wait_per_work * (1.0 + 0.5 * work_density) * (1.0 - is_urgent)
    
    # Normalize components robustly
    norm_inv_slack = robust_mad_norm(inv_slack)
    norm_risk_criticality = robust_mad_norm(risk_criticality)
    norm_energy_density = robust_mad_norm(energy_density)
    norm_wait_penalty = robust_mad_norm(wait_penalty)
    
    # Base score construction: urgency dominates, others contribute additively only when not urgent
    base_score = np.full(N, 1.0, dtype=float)
    score = np.where(is_urgent, -1e12, base_score)
    
    # Convex combination with learned weights (sum = 1.0): urgency-weighted fusion
    # Prioritizes inv_slack (0.35), risk-criticality (0.3), energy (0.2), starvation (0.15)
    non_urgent_contrib = (
        0.35 * norm_inv_slack +
        0.30 * norm_risk_criticality +
        0.20 * (norm_energy_density * energy_penalty_mask) +
        0.15 * norm_wait_penalty
    )
    
    score = np.where(is_urgent, score, score + non_urgent_contrib)
    
    # Final safeguard: ensure finite values and correct shape
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
