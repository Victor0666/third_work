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
    Hybrid priority rule: hard deadline gating + uncertainty-aware slack protection + 
    starvation-robust waiting boost + energy-efficiency dominance only under safety.
    Combines Parent 2's strict DDL-first enforcement and log-scaled waiting with Parent 1's
    risk-sigmoid deadline penalty and normalized criticality weighting, while improving
    numerical stability via IQR clipping and bounded transformations.
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
        """IQR-based normalization clipped to [-3, 3] for stability and outlier resilience"""
        q1 = np.percentile(x, 25)
        q3 = np.percentile(x, 75)
        iqr = q3 - q1 + eps
        normed = (x - q1) / iqr
        return np.clip(normed, -3.0, 3.0)
    
    # Effective slack: multiplicative uncertainty damping preserves sign & relative urgency
    effective_slack = slack / (1.0 + np.clip(uncertainty / 10.0, 0.0, 1000.0))
    
    # Deadline risk: sigmoid penalty for negative slack — smooth, bounded, monotonic
    median_abs_slack = np.abs(np.median(effective_slack)) + eps
    risk_sigmoid = 1.0 / (1.0 + np.exp(-effective_slack / median_abs_slack))
    deadline_risk_raw = np.where(effective_slack < 0, risk_sigmoid, 0.0)
    deadline_risk = robust_normalize(deadline_risk_raw)
    
    # Gated criticality: full upward_rank when safe; linearly increased urgency when at-risk
    # Avoids division-by-zero and preserves ordering under violation
    upward_rank_gated = np.where(
        effective_slack >= 0,
        upward_rank,
        upward_rank * (1.0 + np.clip(-effective_slack, 0.0, 10.0))
    )
    upward_rank_norm = robust_normalize(upward_rank_gated)
    
    # Energy efficiency term: only active when all tasks are deadline-safe
    total_latency = min_exec_time + min_comm_time + eps
    energy_per_latency = min_incremental_energy / total_latency
    energy_eff_norm = robust_normalize(energy_per_latency)
    has_violation = np.any(effective_slack < 0)
    energy_weight = 0.0 if has_violation else 0.85
    
    # Starvation mitigation: log1p-scaled wait time, normalized by median wait
    # Prevents saturation while preserving sensitivity to long waits
    median_wait = np.median(ready_wait_time + eps)
    wait_boost_raw = np.log1p(ready_wait_time / (median_wait + eps))
    wait_boost = robust_normalize(wait_boost_raw)
    
    # Work importance: normalized remaining work supports critical-path progression
    work_norm = robust_normalize(remaining_work)
    
    # Uncertainty modulation: dampens waiting boost only, not deadline or energy terms
    # Ensures high-uncertainty tasks aren't unfairly penalized for urgency or efficiency
    uncertainty_norm = robust_normalize(uncertainty)
    
    # Final score: deadline risk dominates; energy optimized only when safe; waiting prevents starvation
    # Coefficients tuned to reflect hierarchy: DDL > criticality > energy > latency > waiting > uncertainty
    score = (
        +3.5 * deadline_risk          # Strongest penalty for violation risk
        - 2.2 * upward_rank_norm     # Reward critical-path progress when safe; increase urgency when at-risk
        + energy_weight * energy_eff_norm  # Only enabled under full deadline safety
        + 0.5 * robust_normalize(total_latency)  # Light latency pressure to avoid long delays
        + 0.65 * wait_boost          # Balanced starvation mitigation
        + 0.1 * uncertainty_norm     # Minimal penalty for unpredictability (avoids over-penalization)
    )
    
    # Ensure finite, deterministic output
    return np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
