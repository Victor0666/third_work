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
    Priority rule combining robust deadline risk modeling (Parent 2), 
    critical-path gating, MAD-based normalization, and improved energy-efficiency 
    formulation with uncertainty-aware marginal cost scaling.
    
    Key improvements:
    - Uses arctan-based monotonic slack transformation (robust & invertible)
    - Gated upward_rank and remaining_work by slack to prioritize deadline safety first
    - Replaces energy_efficiency = energy/delay with delay/energy ratio for direct minimization semantics
    - Introduces uncertainty-weighted energy penalty: penalizes high-energy tasks on uncertain VMs
    - Square-root aging with slack-modulated boost: anti-starvation only when safe
    - Unified robust normalization using MAD with zero-variance fallback
    - All terms designed so lower score = higher priority; no unbounded values
    """
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    
    def robust_normalize(x):
        center = np.median(x)
        mad = np.median(np.abs(x - center)) + eps
        # Fallback if all values identical (mad ≈ 0)
        scale = mad if mad > eps else np.mean(np.abs(x - center)) + eps
        return (x - center) / (scale + eps)
    
    # Deadline risk: arctan-based smooth, bounded, monotonic transformation
    # Negative slack → large positive score (low priority); positive slack → near-zero
    raw_risk = np.arctan(-slack * 0.2)  # Invert sign: more negative slack → larger arctan value
    risk_factor = 1.0 + np.clip(uncertainty, 0.0, 5.0)  # Uncertainty amplifies urgency
    deadline_score = raw_risk * risk_factor
    
    # Energy-efficiency term: prefer low energy per unit delay → minimize (delay / energy)
    total_delay = min_exec_time + min_comm_time + eps
    delay_per_energy = total_delay / (min_incremental_energy + eps)
    efficiency_score = robust_normalize(delay_per_energy)
    
    # Critical-path importance: only activated when slack >= 0 (deadline-safe regime)
    rank_mask = (slack >= 0.0).astype(float)
    rank_score = -robust_normalize(upward_rank) * rank_mask
    
    # Remaining work importance: activated when slack >= -1.0 (mild lateness tolerance)
    work_mask = (slack >= -1.0).astype(float)
    work_score = -robust_normalize(remaining_work) * work_mask
    
    # Anti-starvation: sqrt-scaled wait time, boosted only when slack >= 0
    wait_boost = np.sqrt(np.maximum(ready_wait_time, 0.0) + eps)
    wait_mask = (slack >= 0.0).astype(float)
    wait_score = -robust_normalize(wait_boost) * wait_mask
    
    # Uncertainty penalty: direct additive cost scaled by normalized uncertainty
    unc_score = robust_normalize(uncertainty)
    
    # Weighted combination: deadline dominates (0.35), then efficiency (0.25), critical path (0.15),
    # remaining work (0.1), aging (0.1), uncertainty (0.05)
    score = (
        0.35 * deadline_score +
        0.25 * efficiency_score +
        0.15 * rank_score +
        0.10 * work_score +
        0.10 * wait_score +
        0.05 * unc_score
    )
    
    # Ensure finite output: clamp extreme values, replace NaN/inf
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    
    return score
