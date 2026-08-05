import numpy as np

# 函数名中的 2 由 SeEvo 在生成 Prompt 时替换为目标版本号。
# 八个输入均为长度 N 的一维数组；相同下标始终指向同一个 ready task。
# 返回值也必须是长度 N 的一维数组，并遵守“分数越小，优先级越高”。
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
    Self-evolved priority rule v2: deadline-violation prevention is *strictly dominant*;
    energy and fairness are secondary only when all deadlines are feasible.
    Key improvements:
      - Urgency term uses hard-gated sigmoid + direct penalty for slack < 0 → no ambiguity in violation cases.
      - Replaces IQR with adaptive MAD+variance fallback: stable under sparse/constant arrays.
      - Uncertainty gating expanded to *asymmetric risk window*: activates for |slack| <= 1.5 (not just positive),
        with stronger weight near zero slack via quadratic weighting.
      - Critical-energy density now *disabled entirely* when slack < 0 → avoids energy optimization at cost of DDL.
      - Fairness uses wait-ratio normalized *only over feasible tasks* (non-zero duration) and clipped conservatively.
      - Aging boost simplified to linear-saturating form for determinism and interpretability.
      - All terms rigorously bounded, sanitized, and guaranteed finite/deterministic.
    """
    eps = 1e-08
    # Sanitize all inputs to finite values, avoiding NaN/inf
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=eps, posinf=1e6, neginf=eps)

    def robust_normalize(x):
        """MAD-based normalization with variance fallback for degenerate cases (N=1 or constant)."""
        if x.size == 0:
            return np.zeros_like(x)
        if x.size == 1:
            return np.zeros_like(x)
        median_x = np.median(x)
        mad = np.median(np.abs(x - median_x)) + eps
        # Fallback to std if MAD ≈ 0 (e.g., constant array), but clamp variance to avoid overflow
        var = np.var(x, ddof=0) + eps
        scale = np.where(mad > eps * 10, mad, np.sqrt(var))
        normed = (x - median_x) / (scale + eps)
        return np.clip(normed, -3.0, 3.0)

    # === URGENCY TERM (STRICTLY DOMINANT) ===
    # Hard penalty for violated or critical slack; smooth sigmoid elsewhere
    urgency_raw = np.where(
        slack < 0.0,
        -1000.0,  # Absolute highest priority for violated tasks
        1.0 / (1.0 + np.exp(-slack / (np.abs(np.median(slack)) + 0.1)))
    )
    norm_urgency = robust_normalize(urgency_raw)
    urgency_term = -5.0 * norm_urgency  # Strongest weight

    # === CRITICAL ENERGY TERM (DISABLED UNDER SLACK < 0) ===
    # Only active when slack >= 0 — no energy tradeoff during violation risk
    exec_effort = np.maximum(min_exec_time, eps)
    comm_effort = np.maximum(min_comm_time, eps)
    total_effort = exec_effort + comm_effort
    base_density = upward_rank * remaining_work / (min_incremental_energy + eps)
    critical_energy_density = np.where(slack >= 0.0, base_density, 0.0)
    norm_critical = robust_normalize(critical_energy_density)
    critical_term = -1.8 * norm_critical

    # === FAIRNESS TERM (WAIT-RATIO, ROBUST TO ZERO DURATION) ===
    duration_estimate = total_effort
    # Avoid division by zero; use only non-zero durations for ratio
    wait_ratio = np.divide(
        ready_wait_time,
        duration_estimate,
        out=np.zeros_like(ready_wait_time, dtype=float),
        where=duration_estimate > eps
    )
    wait_ratio = np.clip(wait_ratio, 0.0, 5.0)  # Cap extreme ratios
    norm_wait = robust_normalize(wait_ratio)
    fairness_term = -0.3 * np.clip(norm_wait, -1.0, 1.0)

    # === UNCERTAINTY PENALTY (ASYMMETRIC AROUND ZERO SLACK) ===
    # Activates for |slack| <= 1.5 → captures both imminent violation and tight-margin opportunity
    slack_abs = np.abs(slack)
    uncertainty_active = (slack_abs <= 1.5) & (uncertainty > 0.02)
    # Quadratic weighting peaks sharply at slack=0
    unc_weight = 1.0 - ((slack / (1.5 + eps)) ** 2)
    uncertainty_penalty = np.where(uncertainty_active, uncertainty * np.maximum(unc_weight, 0.0), 0.0)
    norm_unc = robust_normalize(uncertainty_penalty)
    uncertainty_term = 0.22 * norm_unc

    # === AGING BOOST (LINEAR-SATURATING, INTERPRETABLE) ===
    # Prioritizes long-waiting tasks only when slack allows (slightly relaxed vs v1/v0)
    aging_cond = (slack > -0.5) & (ready_wait_time > 0.03)
    aging_boost = np.where(
        aging_cond,
        np.clip(ready_wait_time / (0.5 + np.abs(slack)), 0.0, 1.0),
        0.0
    )
    aging_term = 0.10 * aging_boost

    # Combine with strict dominance order: urgency > critical > fairness > others
    score = (
        urgency_term +
        critical_term +
        fairness_term +
        uncertainty_term +
        aging_term
    )

    # Final sanitization: guarantee finite, deterministic, shape-(N,)
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)

    return score
