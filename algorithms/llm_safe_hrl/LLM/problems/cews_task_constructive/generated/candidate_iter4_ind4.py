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
    Mutated priority rule: replaces exponential deadline penalty with smooth arctan-based risk,
    introduces critical-energy ratio (not inverse) scaled by slack margin, replaces wait boost
    with percentile-based fairness term bounded by MAD, and integrates uncertainty as multiplicative
    damping factor on energy efficiency under tight slack — all normalized via robust clipped IQR.
    
    Key mutations:
    - Deadline risk uses arctan(-slack) for smooth, bounded, monotonic penalty (no explosion at large negative slack)
    - Critical-energy ratio = (upward_rank * remaining_work) / (min_incremental_energy + eps) — 
      prioritizes high-impact low-energy tasks only when slack > 0; zeroed otherwise.
    - Fairness term: percentile-rank of sqrt(ready_wait_time), clipped to [0, 0.1] — more stable than max-normalization.
    - Uncertainty now acts as a *damping coefficient* on energy efficiency: reduces priority boost when risk is high,
      especially near deadline (scaled by sigmoid of |slack|).
    - All features normalized with same safe_iqr_normalize, but energy-efficiency term uses *ratio-based* scaling
      instead of saturating sigmoid — preserves ordinal ranking fidelity.
    - Weight hierarchy adjusted: deadline (3.5) >> critical-energy (2.0) >> upward_rank (1.2) >> fairness (0.12) >> uncertainty damping (0.0) >> work (0.08)
      — work retained only as tie-breaker; uncertainty contributes only via modulation, not additive score.
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
    
    def safe_iqr_normalize(x):
        """IQR normalization robust to N=1 and constant arrays; returns zeros if degenerate."""
        if x.size == 1:
            return np.zeros_like(x)
        q1 = np.percentile(x, 25)
        q3 = np.percentile(x, 75)
        iqr = q3 - q1 + eps
        if iqr < eps:
            return np.zeros_like(x)
        normed = (x - q1) / iqr
        return np.clip(normed, -3.0, 3.0)
    
    # Smooth, bounded deadline risk: arctan(-slack) → positive penalty for negative slack, zero at slack >= 0
    # arctan range [-pi/2, pi/2]; we shift & scale to [0, ~1.57] for negative slack, 0 otherwise
    deadline_risk_raw = np.where(slack < 0, np.arctan(-slack), 0.0)
    deadline_score = safe_iqr_normalize(deadline_risk_raw)
    
    # Critical-energy ratio: higher upward_rank × work per joule = higher priority (when slack > 0)
    crit_energy_ratio = (upward_rank * remaining_work) / (min_incremental_energy + eps)
    crit_energy_gated = np.where(slack > 0, crit_energy_ratio, 0.0)
    crit_energy_norm = safe_iqr_normalize(crit_energy_gated)
    
    # Upward rank alone (for residual criticality even under tight slack, but gated to avoid conflict with deadline)
    upward_rank_gated = np.where(slack > -eps, upward_rank, 0.0)  # allow tiny margin
    upward_rank_norm = safe_iqr_normalize(upward_rank_gated)
    
    # Fairness: percentile-rank of sqrt(wait), bounded to [0, 0.1]
    wait_sqrt = np.sqrt(np.clip(ready_wait_time, 0.0, None))
    if len(wait_sqrt) > 1:
        # Percentile rank: 0.0 to 1.0, then clip and scale
        wait_rank = np.argsort(np.argsort(wait_sqrt)) / (len(wait_sqrt) - 1 + eps)
        wait_boost = np.clip(wait_rank, 0.0, 1.0) * 0.1
    else:
        wait_boost = np.array([0.0])
    
    # Uncertainty as multiplicative damping on crit_energy_norm when slack is tight
    # Sigmoid of |slack| decays damping as margin grows: high damping near deadline, low far from it
    slack_abs = np.abs(slack)
    damping_factor = 1.0 - 1.0 / (1.0 + np.exp(-(slack_abs - 1.0) / 0.5))  # peaks near slack=0
    uncertainty_damped = crit_energy_norm * (1.0 - uncertainty * damping_factor / (uncertainty.max() + eps))
    
    # Work term: only as mild tie-breaker, normalized
    remaining_work_norm = safe_iqr_normalize(remaining_work)
    
    # Final score: smaller = better. Deadline dominates; critical-energy drives efficiency under feasibility;
    # upward_rank adds backup criticality; fairness prevents starvation; work breaks ties.
    score = (
        +3.5 * deadline_score
        - 2.0 * uncertainty_damped
        - 1.2 * upward_rank_norm
        + 0.12 * wait_boost
        + 0.08 * remaining_work_norm
    )
    
    return np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
