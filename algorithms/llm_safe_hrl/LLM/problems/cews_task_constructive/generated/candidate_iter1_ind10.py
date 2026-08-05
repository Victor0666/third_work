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
    """Novel priority rule emphasizing deadline urgency, risk-aware criticality, 
    and energy-efficiency under feasibility constraints.
    
    Key innovations:
    - Uses *slack-sensitive exponential gating*: sharply penalizes negative slack
      via stable exp(-slack) for urgency (not linear), but clamped to avoid overflow.
    - Introduces *energy-criticality ratio*: normalizes incremental energy against
      upward_rank to favor low-energy execution of high-impact tasks.
    - Replaces linear waiting-time boost with *normalized wait-ratio* relative to
      task's own min_exec_time + min_comm_time, preventing starvation without biasing
      ultra-short tasks.
    - Combines execution + communication into *latency-pressure* term, normalized
      by slack magnitude (with epsilon) to reflect time-sensitivity per unit delay.
    - Applies *uncertainty-weighted energy* instead of raw energy: higher uncertainty
      amplifies energy penalty to discourage risky low-energy assignments.
    - All features robustly normalized using median-IQR (more outlier-resistant than mean-abs).
    """
    eps = 1e-8

    # Safe array conversion without in-place modification
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    # Robust normalization: median + IQR (interquartile range), avoids outlier skew
    def robust_normalize(x):
        q1, q3 = np.percentile(x, [25, 75], method='midpoint')
        iqr = q3 - q1 + eps
        med = np.median(x)
        return (x - med) / iqr

    # === Urgency: exponential slack gating (stable, bounded) ===
    # For slack <= 0: exp(-slack) grows rapidly → high urgency → strong negative contribution
    # For slack > 0: capped at 1.0 → neutral; avoids over-prioritizing far-deadline tasks
    slack_urgency = np.where(
        slack <= 0,
        np.clip(np.exp(-slack), 1.0, 1e4),  # clamp to prevent overflow
        1.0
    )

    # === Criticality-adjusted energy: favors low-energy on high-upward-rank paths ===
    # Avoid division by zero: upward_rank + eps, and use max(eps, ...) for safety
    energy_critical_ratio = min_incremental_energy / (np.maximum(upward_rank, eps) + eps)

    # === Latency pressure: how much "time budget" is consumed per unit of exec+comm ===
    # Use |slack| + eps as denominator → smaller slack → larger pressure weight
    latency_pressure = (min_exec_time + min_comm_time) / (np.abs(slack) + eps)

    # === Risk-amplified energy: uncertainty scales energy cost to discourage unsafe low-energy picks ===
    risk_weighted_energy = min_incremental_energy * (1.0 + uncertainty)

    # === Starvation mitigation: wait ratio relative to intrinsic latency (not absolute time) ===
    intrinsic_latency = min_exec_time + min_comm_time + eps
    wait_ratio = ready_wait_time / intrinsic_latency

    # === Feature assembly: all terms scaled and combined with domain-aligned signs ===
    # Smaller score = higher priority → urgent/negative-slack tasks get large *negative* contribution
    # Energy terms are positive (penalties); wait_ratio is positive (boosts long-waiters)
    score = (
        0.30 * robust_normalize(latency_pressure)       # latency sensitivity
        + 0.25 * robust_normalize(risk_weighted_energy)  # risk-aware energy penalty
        + 0.15 * robust_normalize(energy_critical_ratio)   # criticality efficiency
        - 1.50 * robust_normalize(slack_urgency)           # dominant urgency gate
        + 0.10 * robust_normalize(wait_ratio)              # anti-starvation
        + 0.10 * robust_normalize(uncertainty)             # direct risk signal
    )

    # Final safeguard: replace NaN/inf with finite values, preserving ordering tendency
    return np.nan_to_num(
        score,
        nan=1e12,
        posinf=1e12,
        neginf=-1e12
    )
