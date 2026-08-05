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
    Mutated priority rule emphasizing:
    - Tighter slack-gating via adaptive sigmoid threshold (30th percentile slack)
    - Uncertainty-as-amplifier: modulates *both* energy and urgency, not just energy
    - Bounded multiplicative fusion (no additive score mixing) for monotonic interpretability
    - Wait boost now scaled by *remaining critical path duration*, not raw wait time
    - Robust min-max normalization (not IQR) for cross-seed consistency and outlier resilience
    - Explicit finite-range clipping on all intermediate ratios and final score
    """
    eps = 1e-8
    # Ensure float64 & copy to avoid mutation
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
    
    # Robust min-max normalization with epsilon guard and hard bounds
    def robust_minmax(x):
        x_min, x_max = np.min(x), np.max(x)
        range_val = x_max - x_min + eps
        normalized = (x - x_min) / range_val
        return np.clip(normalized, 0.0, 1.0)
    
    # 1. Adaptive slack gating: sigmoid centered at 30th percentile slack
    slack_thresh = np.quantile(slack, 0.3)  # adaptive threshold
    rel_slack_shifted = (slack - slack_thresh) / (np.abs(slack_thresh) + eps)
    urgency_gate = 1.0 / (1.0 + np.exp(2.0 * rel_slack_shifted))  # [0,1], high when slack <= thresh
    
    # 2. Risk-modulated urgency: amplify urgency under high uncertainty *and* negative slack
    risk_factor = np.clip(1.0 + uncertainty * np.maximum(0.0, -slack) / (np.abs(slack_thresh) + eps), 1.0, 5.0)
    urgency_score = urgency_gate * risk_factor
    
    # 3. Criticality-energy ratio with uncertainty-denominator modulation (not exponent)
    #   Prevents explosion; uses uncertainty as stabilizing divisor
    energy_risk_denom = np.maximum(min_incremental_energy * (1.0 + uncertainty), eps)
    crit_energy_ratio = upward_rank / energy_risk_denom
    # Clip extreme ratios before normalization
    crit_energy_ratio = np.clip(crit_energy_ratio, 1e-6, 1e6)
    crit_energy_norm = robust_minmax(crit_energy_ratio)
    
    # 4. Work-aware waiting boost: only for non-urgent tasks, scaled by *critical path duration*
    #    Critical path duration proxy = upward_rank * (min_exec_time + min_comm_time)
    crit_path_duration = upward_rank * (min_exec_time + min_comm_time + eps)
    max_crit_dur = np.maximum(np.max(crit_path_duration), eps)
    # Boost only if not already urgent (urgency_gate < 0.7) and has meaningful work
    wait_boost_mask = (urgency_gate < 0.7) & (remaining_work > eps)
    norm_wait = np.clip(ready_wait_time / (max_crit_dur + eps), 0.0, 1.0)
    wait_boost = np.where(wait_boost_mask, norm_wait * 0.25, 0.0)
    
    # 5. Energy density term: normalized incremental energy, penalized by uncertainty
    energy_density = min_incremental_energy / (np.maximum(remaining_work, eps))
    energy_density = np.clip(energy_density, eps, 1e6)
    energy_norm = robust_minmax(energy_density)
    
    # 6. Uncertainty penalty: bounded and normalized
    unc_norm = robust_minmax(uncertainty)
    
    # Multiplicative fusion: urgency dominates, others refine within its envelope
    # All terms in [0,1] → product preserves ordering and avoids sign flips
    base_priority = (
        (1.0 - urgency_score) *  # smaller urgency_score → higher priority (since we minimize)
        (1.0 - 0.5 * crit_energy_norm) *
        (1.0 + 0.3 * energy_norm) *
        (1.0 + 0.2 * unc_norm) *
        (1.0 - 0.25 * wait_boost)
    )
    
    # Final score: ensure finite, deterministic, shape-(N,)
    score = np.clip(base_priority, 1e-12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=1e12)
    
    return score
