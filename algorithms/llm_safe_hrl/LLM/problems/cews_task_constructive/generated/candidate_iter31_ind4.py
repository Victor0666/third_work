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
    v2: SEER-Adaptive Criticality-Gated Energy-Risk Priority (SCGERP)
    - Replaces piecewise urgency with smooth, monotonic arctan-based urgency scaling
      that is numerically stable and zero-crossing at slack=0.
    - Introduces *critical path density* (cp_density) as a normalized bottleneck signal,
      scaled only when slack > tau_safe to avoid interfering with hard DDL enforcement.
    - Uses IQR-based robust normalization (not min-max) for better outlier resilience.
    - Couples uncertainty *multiplicatively* into energy term *only under safety margin*,
      and into fairness term *only under low-latency regimes* (slack > 0.5 * tau_safe).
    - Adds starvation relief via *relative wait ratio*: ready_wait_time / (min_exec_time + min_comm_time + eps),
      capped and normalized — rewards tasks waiting long relative to their latency footprint.
    - All terms are sign-consistent: lower score = higher priority; violation override preserves DDL dominance.
    - No exponentials, no log(0), no tanh saturation; all ops epsilon-guarded and bounded.
    """
    eps = 1e-8
    tau_safe = np.quantile(np.abs(slack), 0.75) + eps  # adaptive safety threshold
    tau_urgency = np.quantile(np.abs(slack[slack != 0]), 0.5) + eps if np.any(slack != 0) else 1.0

    def sanitize(x):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(x, nan=eps, posinf=1e6, neginf=-1e6)

    min_exec_time = sanitize(min_exec_time)
    min_comm_time = sanitize(min_comm_time)
    min_incremental_energy = sanitize(min_incremental_energy)
    slack = sanitize(slack)
    upward_rank = sanitize(upward_rank)
    remaining_work = sanitize(remaining_work)
    ready_wait_time = sanitize(ready_wait_time)
    uncertainty = sanitize(uncertainty)

    # Robust slack: subtract uncertainty only when slack > 0; preserve negative slack integrity
    robust_slack = np.where(slack > 0, slack - uncertainty, slack)

    # Smooth, monotonic urgency: arctan-based, zero at slack=0, bounded in [-π/2, π/2], inverted for priority
    # Ensures continuity, avoids clipping artifacts, and remains well-behaved near zero
    urgency_raw = -np.arctan(robust_slack / (tau_urgency + eps))

    # Latency pressure: criticality-weighted latency burden, attenuated by uncertainty and gated by urgency
    exec_comm_sum = min_exec_time + min_comm_time + eps
    latency_pressure = (upward_rank + eps) * exec_comm_sum / (tau_urgency + eps + np.abs(robust_slack))
    latency_pressure = latency_pressure * (1.0 - np.clip(uncertainty, 0.0, 1.0))
    # Gate with urgency magnitude: amplify pressure when urgency > 0.3 (i.e., slack < ~0.3*tau_urgency)
    urgency_gate = np.clip(1.0 + 2.0 * np.abs(urgency_raw) / (np.pi/2), 1.0, 3.0)
    latency_pressure *= urgency_gate

    # Safety-gated energy term: only active when robust_slack > 0, scaled by uncertainty *only* then
    safety_mask = robust_slack > eps
    energy_term = np.where(
        safety_mask,
        min_incremental_energy / (exec_comm_sum + 1.0) * (1.0 + uncertainty * 0.3),
        min_incremental_energy / (exec_comm_sum + 1.0) * 2.0  # higher penalty when unsafe
    )

    # Critical path density: bottleneck intensity (work × criticality / latency), activated only in safe regime
    cp_density = (upward_rank + eps) * (remaining_work + eps) / (exec_comm_sum + eps)
    cp_gated = np.where(safety_mask, cp_density * (1.0 - np.clip(uncertainty, 0.0, 1.0)), 0.0)

    # Fairness: two components — (1) relative wait ratio (anti-starvation), (2) uncertainty-boosted wait under safety
    rel_wait_ratio = np.clip(ready_wait_time / (exec_comm_sum + eps), 0.0, 100.0)
    fairness_base = rel_wait_ratio
    fairness_boost = np.where(
        robust_slack > 0.5 * tau_safe,
        uncertainty * 0.8 * ready_wait_time,
        0.0
    )
    fairness_term = fairness_base + fairness_boost

    # Robust IQR-based normalization (resilient to outliers)
    def robust_normalize(x):
        x = np.clip(x, -1e6, 1e6)
        q25, q75 = np.percentile(x, [25, 75]) if x.size > 2 else (np.min(x), np.max(x))
        iqr = q75 - q25 + eps
        center = (q25 + q75) / 2.0
        norm = (x - center) / iqr
        # Clamp normalized values to [-3, 3] to prevent extreme weights
        return np.clip(norm, -3.0, 3.0)

    norm_urgency = robust_normalize(urgency_raw)
    norm_latency = robust_normalize(latency_pressure)
    norm_energy = robust_normalize(energy_term)
    norm_cp = robust_normalize(cp_gated)
    norm_fair = robust_normalize(fairness_term)

    # Weighted linear combination: urgency and latency dominate; energy secondary; fairness & cp minor
    # Negative weights for urgency/latency/energy (lower = better); positive for cp/fairness (higher = better → flipped sign)
    score = (
        -15.0 * norm_urgency
        - 10.0 * norm_latency
        - 7.0 * norm_energy
        + 2.0 * norm_cp   # prioritize bottleneck relief *only* when safe
        - 1.5 * norm_fair  # fairness lowers priority (longer wait → higher score → lower priority unless compensated)
    )

    # Hard deadline override: any task with slack < -eps gets *lowest possible score* (highest priority)
    violation_mask = slack < -eps
    if np.any(violation_mask):
        base_ref = np.min(score[~violation_mask]) if np.any(~violation_mask) else np.min(score)
        score = np.where(violation_mask, base_ref - 1e9, score)

    # Final sanitization: ensure finite, shape-(N,)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)
    return score.reshape(-1)
