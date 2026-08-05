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
    - Uses *inverse slack* (not relative slack) for urgency, clipped to [0.1, 100] to avoid instability near zero
    - Introduces 'risk-adjusted criticality' = upward_rank * (1 + uncertainty), gated by slack < median_slack
    - Energy penalty now scaled by *both* criticality and risk-adjusted duration (exec+comm)*(1+uncertainty)
    - Starvation relief uses wait_per_work *and* normalized wait_time, but only activated when work density > median
    - All components normalized via robust MAD scaling; no min-max fallback degeneracy
    - Urgent tasks (slack <= 0) assigned -1e15 (not -1e12) for stronger hard-constraint enforcement
    - Final score combines urgency, critical-energy, fairness, and risk terms with adaptive weights summing to 1.0
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

    def robust_mad_norm(x):
        if x.size == 0:
            return np.zeros_like(x)
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        scale = mad * 3.0 + eps
        normed = (x - med) / scale
        # Clip extreme outliers but preserve rank order
        return np.clip(normed, -3.0, 3.0)

    # Hard urgency override: strict deadline enforcement
    is_urgent = (slack <= 0.0).astype(float)
    
    # Absolute urgency term: inverse slack, stable near zero
    inv_slack = np.divide(1.0, np.abs(slack) + eps, out=np.zeros_like(slack), where=np.abs(slack) + eps != 0)
    inv_slack = np.clip(inv_slack, 0.1, 100.0)
    norm_inv_slack = robust_mad_norm(inv_slack)
    
    # Risk-adjusted critical path: upward_rank amplified by uncertainty, active only under pressure
    median_slack = np.median(slack) if N > 1 else slack[0]
    pressure_mask = (slack < median_slack).astype(float)
    risk_adjusted_rank = upward_rank * (1.0 + uncertainty)
    norm_risk_rank = robust_mad_norm(risk_adjusted_rank) * pressure_mask
    
    # Risk-adjusted duration for energy grounding
    duration = min_exec_time + min_comm_time + eps
    risk_duration = duration * (1.0 + uncertainty)
    
    # Energy density: marginal energy per risk-adjusted duration
    energy_density = np.divide(min_incremental_energy, risk_duration, out=np.zeros_like(min_incremental_energy), where=risk_duration != 0)
    norm_energy_density = robust_mad_norm(energy_density)
    
    # Critical-energy coupling: only penalize high-energy tasks on high-risk critical paths
    critical_energy_penalty = norm_energy_density * norm_risk_rank * (1.0 + uncertainty)
    
    # Fairness: starvation relief gated by workload density
    wait_per_work = np.divide(ready_wait_time, remaining_work + eps, out=np.zeros_like(ready_wait_time), where=remaining_work + eps != 0)
    median_work = np.median(remaining_work) if N > 1 else remaining_work[0]
    work_density_gate = (remaining_work > median_work).astype(float)
    norm_wait_time = robust_mad_norm(ready_wait_time)
    fairness_term = norm_wait_time * wait_per_work * work_density_gate
    
    # Uncertainty-weighted urgency amplification
    uncertainty_boost = uncertainty * norm_inv_slack * norm_risk_rank
    
    # Base priority composition (weights sum to 1.0)
    # Higher weight on urgency & critical-energy to enforce DDL + energy tradeoff
    score = np.full(N, 0.0, dtype=float)
    score = np.where(is_urgent, -1e15, score)
    
    # Non-urgent branch: weighted linear combination
    score = np.where(
        is_urgent,
        score,
        (0.42 * norm_inv_slack) +
        (0.30 * critical_energy_penalty) +
        (0.18 * fairness_term) +
        (0.07 * robust_mad_norm(uncertainty_boost)) +
        (0.03 * robust_mad_norm(remaining_work))
    )
    
    # Final safeguard: finite values only
    score = np.nan_to_num(score, nan=1e15, posinf=1e15, neginf=-1e15)
    score = np.clip(score, -1e15, 1e15)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
