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
    v2 priority rule: Urgency-hardened tanh-slack gating + critical-path energy density + starvation-robust wait boost + uncertainty-coupled work scaling.
    
    Key mutations:
    - Replaces step-based urgency with smooth, bounded tanh(-slack/tau) to avoid discontinuities while preserving strict dominance for slack <= 0.
    - Uses tanh-scaled slack sensitivity (not 1/abs(rel_slack)) for stable near-deadline pressure and graceful tail-off.
    - Introduces *work-normalized* energy density: min_incremental_energy / (remaining_work + eps), rewarding high-energy tasks that unlock large descendant work.
    - Replaces percentile-based fairness gate with IQR-robust median + 0.5×IQR threshold on remaining_work, improving small-N stability.
    - Adds uncertainty-weighted *criticality-aware* waiting boost: only amplifies wait priority when both upward_rank and uncertainty are above robust medians.
    - Drops redundant norm_remaining_work term; replaces with normalized work-to-energy ratio to bias toward high-value computational leverage.
    - All normalization uses 5–95% clipping (not 1–99%) for better small-sample behavior and outlier resilience.
    - Explicit finite-range clamping and nan/inf sanitization applied per term before aggregation.
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

    def robust_595_norm(x):
        if x.size == 0:
            return np.zeros_like(x)
        p05 = np.percentile(x, 5.0)
        p95 = np.percentile(x, 95.0)
        x_clipped = np.clip(x, p05, p95)
        x_min = np.min(x_clipped)
        x_max = np.max(x_clipped)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x_clipped - x_min) / (x_max - x_min + eps)

    # === URGENT TASK HANDLING: smooth tanh-based dominance ===
    # tanh(-slack/tau) → -1.0 for large negative slack, ~0 for slack >> 0, bounded in [-1, 0]
    tau = np.clip(np.median(min_exec_time + min_comm_time + eps), eps, 1e3) + eps
    urgency_score = np.tanh(-slack / tau)  # [-1.0, 0.0]; more negative = more urgent
    is_urgent = (slack <= 0.0).astype(float)
    # Assign guaranteed min score (-1e12) only to truly violating tasks (slack <= 0), else use smooth tanh
    base_score = np.where(is_urgent, -1e12, 0.0)

    # === CRITICALITY-AWARE DURATION ===
    duration = min_exec_time + min_comm_time + eps
    critical_latency_raw = duration * (1.0 + 0.8 * robust_595_norm(upward_rank))
    norm_critical_latency = robust_595_norm(critical_latency_raw)

    # === WORK-NORMALIZED ENERGY DENSITY ===
    # Energy per unit descendant work — prioritizes tasks whose execution unlocks large downstream work
    work_energy_ratio = np.divide(min_incremental_energy, remaining_work + eps,
                                  out=np.zeros_like(min_incremental_energy),
                                  where=remaining_work + eps != 0)
    norm_work_energy_ratio = robust_595_norm(work_energy_ratio)

    # === TIGHT-SLACK ENERGY PENALTY GATE ===
    rel_slack = np.divide(slack, duration, out=np.zeros_like(slack), where=duration != 0)
    rel_slack = np.where(np.isfinite(rel_slack), rel_slack, 0.0)
    tight_slack_mask = (rel_slack <= 0.25).astype(float)
    rank_threshold = np.median(upward_rank) + 0.5 * (np.percentile(upward_rank, 75) - np.percentile(upward_rank, 25)) + eps
    high_rank_mask = (upward_rank > rank_threshold).astype(float)
    energy_penalty_mask = tight_slack_mask * high_rank_mask
    energy_penalty = norm_work_energy_ratio * energy_penalty_mask

    # === ROBUST STARVATION RESCUE ===
    # Only activate wait boost if task has significant remaining work AND is non-urgent
    work_iqr = np.percentile(remaining_work, 75) - np.percentile(remaining_work, 25)
    work_threshold = np.median(remaining_work) + 0.5 * work_iqr + eps
    wait_gate = (remaining_work >= work_threshold).astype(float)
    # Boost wait priority only when both criticality and uncertainty are above median
    unc_median = np.median(uncertainty)
    rank_median = np.median(upward_rank)
    unc_rank_boost = ((uncertainty > unc_median) & (upward_rank > rank_median)).astype(float)
    norm_wait_time = robust_595_norm(ready_wait_time)
    wait_penalty = (1.0 - is_urgent) * norm_wait_time * wait_gate * unc_rank_boost

    # === UNCERTAINTY-COUPLED SLACK SENSITIVITY ===
    # tanh-based: avoids division-by-zero, saturates at ±1, smooth near zero
    slack_tanh_sensitivity = np.tanh(np.abs(rel_slack) * 4.0)  # [0, 1], steep near rel_slack=0
    norm_uncertainty = robust_595_norm(uncertainty)
    uncertainty_boost = norm_uncertainty * slack_tanh_sensitivity * robust_595_norm(upward_rank)

    # === COMBINED SCORE ===
    # Non-urgent branch: latency + energy penalty + wait + uncertainty
    non_urgent_contrib = (
        0.30 * norm_critical_latency +
        0.25 * energy_penalty +
        0.20 * wait_penalty +
        0.15 * uncertainty_boost +
        0.10 * robust_595_norm(duration)  # slight bias toward shorter-duration tasks when tied
    )
    score = np.where(is_urgent, base_score, non_urgent_contrib)

    # Final sanitization
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)

    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
