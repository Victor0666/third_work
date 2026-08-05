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
    v2: Hybrid priority rule combining Parent 2's strict deadline gating and MAD fidelity
         with Parent 1's robust arctan urgency (smooth, bounded, stable) and critical-energy density,
         plus novel latency-risk modulation and fairness-aware starvation prevention.

    Key innovations:
    - Dual-mode urgency: arctan-based for slack >= 0 (smooth, bounded sensitivity), exponential penalty for slack < 0 (hard violation dominance)
    - Critical-energy density (CED) gated *only* when slack > 0, normalized via MAD to preserve weight hierarchy
    - Latency inflation now risk-adaptive: applies uncertainty only when slack <= 0 *and* uncertainty > 0.1 → avoids over-penalizing low-risk tasks
    - Fairness term uses sqrt(wait)/median_wait clipped to [0, 0.15], applied unconditionally but down-weighted (0.08) to prevent starvation without compromising DDL
    - Unified safe_mad_normalize with degenerate fallback (N=1 → zeros) and hard clipping [-4.0, 4.0]
    - Final weights enforce objective priority: deadline (4.5) >> CED (2.4) >> upward_rank (1.4) >> latency-inflation (0.35) >> fairness (0.08)
    - All operations guarded against NaN/inf/zero via eps = 1e-8 and nan_to_num
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

    def safe_mad_normalize(x):
        """MAD normalization robust to N=1, constants, and outliers; returns clipped [-4,4] values."""
        if x.size == 1:
            return np.zeros_like(x)
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + eps
        if mad < eps:
            return np.zeros_like(x)
        normed = (x - med) / mad
        return np.clip(normed, -4.0, 4.0)

    # === Urgency: arctan ramp for slack >= 0, exponential penalty for slack < 0 ===
    # Ensures monotonicity and avoids gradient dilution near zero while preserving smoothness
    arctan_urgency = 0.5 + (1.0 / np.pi) * np.arctan(np.where(slack >= 0, slack, 0.0) / 5.0)
    exp_violation = np.where(slack < -eps, np.exp(np.clip(-slack, 0.0, 25.0)) - 1.0, 0.0)
    deadline_risk_raw = arctan_urgency + exp_violation
    deadline_score = safe_mad_normalize(deadline_risk_raw)

    # === Critical-energy density: "critical work per joule", active only when slack > 0 ===
    ced_numerator = upward_rank * remaining_work + eps
    ced_denominator = min_incremental_energy + eps
    ced_raw = np.where(slack > 0, ced_numerator / ced_denominator, 0.0)
    ced_norm = safe_mad_normalize(ced_raw)

    # === Upward rank fidelity: only matters when slack > 0 (critical-path relevance) ===
    upward_rank_active = np.where(slack > 0, upward_rank, 0.0)
    upward_rank_norm = safe_mad_normalize(upward_rank_active)

    # === Risk-weighted latency inflation: only under tight margin AND significant uncertainty ===
    base_latency = min_exec_time + min_comm_time + eps
    inflated_latency = np.where(
        (slack <= 0) & (uncertainty > 0.1),
        base_latency * (1.0 + np.clip(uncertainty, 0.0, 0.6)),
        base_latency
    )
    latency_inflation_raw = np.where(slack <= 0, inflated_latency, 0.0)
    latency_inflation_norm = safe_mad_normalize(latency_inflation_raw)

    # === Fairness: sqrt-scaled wait time to prevent starvation, always active but low weight ===
    median_wait = np.median(ready_wait_time) + eps
    rel_sqrt_wait = np.sqrt(np.maximum(ready_wait_time, 0.0)) / median_wait
    fairness_boost = np.clip(rel_sqrt_wait, 0.0, 0.15)

    # === Final weighted score: smaller = higher priority ===
    score = (
        +4.5 * deadline_score
        - 2.4 * ced_norm
        - 1.4 * upward_rank_norm
        + 0.35 * latency_inflation_norm
        + 0.08 * fairness_boost
    )

    # Ensure finite, deterministic output
    return np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
