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
    """Novel priority rule emphasizing deadline safety via adaptive risk scaling,
    critical-path awareness with workload-normalized urgency, and starvation-avoiding
    waiting-time modulation — all under robust per-feature normalization.

    Key innovations:
      - Deadline risk is modeled as *exponential penalty* only for negative slack,
        scaled by normalized upward_rank to amplify urgency on critical paths.
      - Energy-awareness uses *energy-per-work* ratio (joules per MI), normalized,
        capturing efficiency rather than raw energy.
      - Communication+execution coupling: uses harmonic mean of min_exec_time and
        min_comm_time to jointly penalize high-latency bottlenecks.
      - Ready wait time contributes *asymmetrically*: linear boost only after threshold
        (to avoid premature promotion), capped to prevent dominance.
      - Uncertainty is folded into a *risk-adjusted slack*, shrinking effective slack
        for high-uncertainty tasks before risk calculation.
      - All features normalized by interquartile range (IQR) + eps for robustness
        against outliers (vs. mean-abs in v1).
    """
    eps = 1e-8

    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    # Robust IQR-based normalization: less sensitive to outliers than mean-abs
    def normalize_iqr(x):
        q75, q25 = np.percentile(x, [75, 25], method='midpoint')
        iqr = q75 - q25
        return x / (iqr + eps)

    # Risk-adjusted slack: shrink slack for high-uncertainty tasks → tighter effective deadline
    # Clamp uncertainty to avoid extreme shrinkage; use soft upper bound
    clipped_uncertainty = np.clip(uncertainty, 0.0, 10.0)
    risk_adjusted_slack = slack - 0.5 * clipped_uncertainty

    # Deadline risk: exponential penalty only when risk_adjusted_slack < 0
    # Prevent overflow: cap exponent at -20 (≈2e-9) → still monotonic & finite
    raw_risk = np.where(risk_adjusted_slack < 0,
                        np.exp(np.clip(-risk_adjusted_slack, 0.0, 20.0)),
                        0.0)

    # Critical-path amplification: multiply risk by upward_rank (higher rank = more critical)
    # Normalize upward_rank first to avoid scale explosion
    norm_upward = normalize_iqr(upward_rank)
    amplified_risk = raw_risk * (1.0 + 0.5 * np.clip(norm_upward, 0.0, np.inf))

    # Energy efficiency signal: incremental energy per unit remaining work (J/MI)
    # Avoid division by zero; replace zeros in remaining_work with small positive value
    safe_remaining_work = np.where(remaining_work == 0.0, eps, remaining_work)
    energy_per_work = min_incremental_energy / safe_remaining_work

    # Harmonic mean of exec & comm time → penalizes tasks where *both* are large
    # Harmonic mean = 2ab/(a+b); defined only if both > 0 → clamp inputs to eps
    a = np.clip(min_exec_time, eps, None)
    b = np.clip(min_comm_time, eps, None)
    harmonic_latency = 2.0 * a * b / (a + b)

    # Starvation-avoiding wait time: linear boost only after 0.5s, capped at 2.0x effect
    wait_boost = np.clip(0.5 * (ready_wait_time - 0.5), 0.0, 2.0)

    # Normalize all components separately
    norm_harmonic = normalize_iqr(harmonic_latency)
    norm_energy_per_work = normalize_iqr(energy_per_work)
    norm_amplified_risk = normalize_iqr(amplified_risk)
    norm_wait_boost = normalize_iqr(wait_boost)

    # Final score: sum of normalized, signed contributions
    # Negative coefficients → higher urgency/risk → lower score (better priority)
    score = (
        + 0.30 * norm_harmonic           # latency penalty
        + 0.35 * norm_energy_per_work   # inefficiency penalty
        - 3.00 * norm_amplified_risk    # urgent deadline risk (strongest weight)
        + 0.10 * norm_wait_boost        # anti-starvation boost (positive → lowers score)
        + 0.05 * normalize_iqr(remaining_work)  # heavier work gets slight penalty (load balance)
        + 0.05 * normalize_iqr(uncertainty)     # high uncertainty slightly penalized
    )

    # Ensure finite output; avoid NaN/inf from any intermediate step
    return np.nan_to_num(
        score,
        nan=1e12,
        posinf=1e12,
        neginf=-1e12
    )
