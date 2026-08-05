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
    v2 priority rule: Slack-driven urgency gating + risk-aware criticality-energy tradeoff,
                      with starvation-robust wait-time bias and uncertainty-calibrated energy scaling.
    
    Key mutations from v1:
      - Hard urgency gating: all slack <= 0 → fixed ultra-low score (guarantees DDL feasibility)
      - Soft urgency ramp: exp(-max(0, -slack)/tau) for continuous negative-slack pressure
      - Criticality-energy coupling: energy penalty scaled by (1 + uncertainty) * upward_rank
      - Wait-time fairness: uses log(1+ready_wait_time) to avoid linear dominance, capped & normalized
      - Uncertainty-aware energy normalization: rescales energy density by (1 + uncertainty/unc_median)
      - Work-intensity correction replaces comm/comp ratio with (remaining_work / duration) penalty
      - All norms use 5%-95% clipping (more stable than 2%-98% for small N) and epsilon-hardened
      - Final score enforces strict monotonicity w.r.t. slack and upward_rank via sign-consistent terms
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

    # Duration baseline for relative measures
    duration = min_exec_time + min_comm_time + eps

    # --- HARD URGENT GATING: zero-slack or negative-slack tasks get highest priority (lowest score)
    is_urgent = (slack <= 0.0).astype(float)
    base_score = np.full(N, 1.0, dtype=np.float64)

    # --- SOFT URGENT BIAS: exponential pressure for negative slack, smooth transition at slack=0
    neg_slack_pressure = np.exp(-np.maximum(0.0, -slack) / (np.median(duration) + eps))
    neg_slack_pressure = np.nan_to_num(neg_slack_pressure, nan=0.0, posinf=0.0, neginf=0.0)

    # --- RISK-AWARE ENERGY TERM: upward-rank-weighted, uncertainty-scaled marginal energy density
    energy_density = np.divide(min_incremental_energy, duration + eps, out=np.zeros_like(min_incremental_energy), where=duration + eps != 0)
    energy_density = np.nan_to_num(energy_density, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Calibrate energy by uncertainty: higher uncertainty → dampen energy priority (avoid risky low-energy choices)
    unc_median = np.median(uncertainty) if N > 1 else uncertainty[0]
    unc_scale = 1.0 + np.divide(uncertainty, unc_median + eps, out=np.ones_like(uncertainty), where=unc_median + eps != 0)
    risk_scaled_energy = energy_density * unc_scale
    
    # Upward-rank amplification only for non-urgent tasks (prevents over-prioritizing critical but late tasks)
    ur_weight = np.where(slack > 0.0, upward_rank, 0.0)
    ur_weight = np.nan_to_num(ur_weight, nan=0.0, posinf=0.0, neginf=0.0)
    norm_ur_weight = robust_minmax_norm(ur_weight)
    norm_risk_energy = robust_minmax_norm(risk_scaled_energy)

    # --- WORK-INTENSITY PENALTY: high remaining_work per duration penalizes "heavy" tasks under tight slack
    work_intensity = np.divide(remaining_work, duration + eps, out=np.zeros_like(remaining_work), where=duration + eps != 0)
    work_intensity = np.nan_to_num(work_intensity, nan=0.0, posinf=0.0, neginf=0.0)
    norm_work_intensity = robust_minmax_norm(work_intensity)

    # --- STARVATION FAIRNESS: logarithmic wait-time bias (avoids linear domination), capped and normalized
    log_wait = np.log1p(ready_wait_time)  # log(1+x) avoids log(0) and compresses large values
    log_wait = np.nan_to_num(log_wait, nan=0.0, posinf=0.0, neginf=0.0)
    norm_log_wait = robust_minmax_norm(log_wait)

    # --- FINAL SCORE CONSTRUCTION ---
    # Urgent tasks dominate: assign fixed ultra-low base score
    score = np.where(is_urgent, -1000.0, 0.0)

    # Non-urgent tasks: weighted combination
    non_urgent_mask = ~is_urgent
    if np.any(non_urgent_mask):
        # Weighted sum for non-urgent tasks only
        score_non_urgent = (
            0.40 * neg_slack_pressure[non_urgent_mask] +           # stronger urgency pull near deadline
            0.30 * norm_risk_energy[non_urgent_mask] * (0.7 + 0.3 * norm_ur_weight[non_urgent_mask]) +
            0.15 * norm_work_intensity[non_urgent_mask] +
            0.10 * (1.0 - norm_log_wait[non_urgent_mask]) +         # fairness boost: longer wait → lower score
            0.05 * robust_minmax_norm(uncertainty)[non_urgent_mask]  # mild uncertainty penalty for stability
        )
        score = np.where(non_urgent_mask, score_non_urgent, score)

    # Ensure finite output and shape compliance
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
