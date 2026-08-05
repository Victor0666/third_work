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
    v2 priority rule: Deadline-hardened urgency-energy-criticality fusion with adaptive risk gating,
                      starvation-robust waiting bias, and normalized work-intensity penalty.
    
    Key mutations from v1:
      - Replaces soft saturation urgency with hard deadline dominance: all slack <= 0 get fixed ultra-low score (0.0)
      - Introduces *dual-gated energy optimization*: only applies energy discount when (slack > 0) AND (uncertainty < median_uncertainty)
        — prioritizes low-risk energy savings over high-risk ones
      - Uses *relative remaining_work density* (remaining_work / duration) instead of raw remaining_work to penalize compute-heavy bottlenecks
      - Replaces fairness_boost_mask with *continuous wait-pressure term*: exp(-max(0, -ready_wait_time/duration)) → grows smoothly for long waits
      - Removes comm/comp ratio penalty; instead uses *normalized communication overhead* as standalone penalty term
      - All normalization now uses 5%-95% percentile clipping (more robust for small-N and skewed distributions)
      - Final score bounded strictly in [0.0, 1.0] to guarantee monotonic interpretability and avoid overflow amplification
      - Explicit NaN/inf protection on every intermediate array before norm
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
        p05 = np.percentile(x, 5.0, method='lower')
        p95 = np.percentile(x, 95.0, method='higher')
        x_clipped = np.clip(x, p05, p95)
        x_min = np.min(x_clipped)
        x_max = np.max(x_clipped)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x_clipped - x_min) / (x_max - x_min + eps)

    # Compute base duration (execution + communication), guard against zero
    duration = min_exec_time + min_comm_time + eps
    duration = np.nan_to_num(duration, nan=eps, posinf=1e12, neginf=eps)

    # Hard deadline dominance: assign minimum priority (0.0) to all overdue or at-risk tasks
    is_overdue = slack <= 0.0
    base_score = np.full(N, 1.0, dtype=np.float64)
    base_score[is_overdue] = 0.0

    # Relative slack: avoid division by zero and clamp extreme values
    rel_slack = np.divide(slack, duration, out=np.zeros_like(slack), where=duration != 0)
    rel_slack = np.nan_to_num(rel_slack, nan=0.0, posinf=0.0, neginf=-1e6)

    # Urgency term: linear decay for positive slack, zero for negative (handled above)
    urgency_term = np.clip(1.0 - np.maximum(0.0, rel_slack), 0.0, 1.0)
    norm_urgency = robust_minmax_norm(urgency_term)

    # Energy term: risk-adjusted marginal energy per time unit
    energy_density = np.divide(min_incremental_energy, duration, out=np.zeros_like(min_incremental_energy), where=duration != 0)
    energy_density = np.nan_to_num(energy_density, nan=0.0, posinf=1e12, neginf=0.0)
    
    # Dual-gated energy discount: only reward energy efficiency when slack > 0 AND uncertainty is *low*
    unc_median = np.median(uncertainty) if N > 1 else uncertainty[0]
    energy_gate = (slack > 0.0) & (uncertainty < unc_median)
    norm_energy = robust_minmax_norm(energy_density)
    energy_discount = -0.25 * norm_energy * energy_gate.astype(float)

    # Criticality: upward rank normalized and weighted by slack safety
    norm_upward_rank = robust_minmax_norm(upward_rank)
    criticality_weight = np.where(slack > 0.0, 0.7 + 0.3 * norm_urgency, 0.0)
    criticality_term = norm_upward_rank * criticality_weight

    # Work intensity: remaining_work density (MI/sec) — penalizes long critical paths with high compute load
    work_density = np.divide(remaining_work, duration, out=np.zeros_like(remaining_work), where=duration != 0)
    work_density = np.nan_to_num(work_density, nan=0.0, posinf=1e12, neginf=0.0)
    norm_work_density = robust_minmax_norm(work_density)

    # Communication overhead penalty: normalized min_comm_time fraction of total duration
    comm_fraction = np.divide(min_comm_time, duration, out=np.zeros_like(min_comm_time), where=duration != 0)
    comm_fraction = np.nan_to_num(comm_fraction, nan=0.0, posinf=1.0, neginf=0.0)
    norm_comm_frac = robust_minmax_norm(comm_fraction)

    # Starvation mitigation: smooth exponential wait pressure (grows continuously with relative wait)
    wait_pressure = np.exp(-np.maximum(0.0, -ready_wait_time / (duration + eps)))
    wait_pressure = np.nan_to_num(wait_pressure, nan=0.0, posinf=0.0, neginf=0.0)
    # Invert to increase priority with longer wait: 1.0 - wait_pressure → 0.0 for fresh, ~1.0 for stale
    wait_bias = 1.0 - wait_pressure
    norm_wait_bias = robust_minmax_norm(wait_bias)

    # Assemble final score: lower is better; hard-deadline tasks already set to 0.0
    # Weighting balances urgency (0.35), energy-efficiency (0.25), criticality (0.20), work density (0.10),
    #   comm overhead (0.05), and wait bias (0.05)
    score = (
        0.35 * norm_urgency +
        0.25 * (norm_energy + energy_discount) +
        0.20 * criticality_term +
        0.10 * norm_work_density +
        0.05 * norm_comm_frac +
        0.05 * norm_wait_bias
    )

    # Apply hard deadline override: ensure all overdue tasks retain absolute top priority (score = 0.0)
    score = np.where(is_overdue, 0.0, score)

    # Final bounds and sanitization
    score = np.clip(score, 0.0, 1.0)
    score = np.nan_to_num(score, nan=1.0, posinf=1.0, neginf=0.0)

    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
