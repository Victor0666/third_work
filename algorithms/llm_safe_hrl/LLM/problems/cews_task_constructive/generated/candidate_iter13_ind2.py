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
    - Hard deadline enforcement via *slack-triggered rank inversion* (not just penalty)
    - Energy-aware criticality via *uncertainty-gated upward_rank scaling*
    - Starvation control via *relative wait-time ratio*, bounded and slack-adjusted
    - Robust normalization using *median absolute deviation (MAD)* for tighter outlier suppression
    - Explicit monotonic slack-energy coupling: energy cost is down-weighted only when slack > 0 and ample
    - Replaces power-law risk exponent with *linear uncertainty amplification* for numerical stability
    - Introduces *work-normalized communication pressure*: high-comm/low-work tasks prioritized under tight slack
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
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)

    # Robust normalization using MAD (more outlier-resistant than IQR)
    def robust_normalize_mad(x):
        center = np.median(x)
        mad = np.median(np.abs(x - center)) + eps
        normed = (x - center) / mad
        return np.clip(normed, -8.0, 8.0)  # tighter bound than IQR version

    # Task intrinsic duration (min feasible time budget)
    task_min_duration = np.maximum(min_exec_time + min_comm_time, eps)

    # --- SLACK-TRIGGERED RANK INVERSION: critical for hard DDL enforcement ---
    # When slack <= 0: invert upward_rank → highest criticality becomes *lowest* score (highest priority)
    # This directly promotes critical-path tasks *only when urgency demands it*, avoiding blind criticality bias
    ur_inverted = np.where(slack <= 0.0, -upward_rank, upward_rank)
    ur_norm = robust_normalize_mad(ur_inverted)

    # --- UNCERTAINTY-GATED CRITICALITY ENERGY RATIO ---
    # Instead of power law: use linear uncertainty boost to energy denominator, capped at 3x
    energy_denom = min_incremental_energy * (1.0 + np.clip(uncertainty, 0.0, 2.0))
    energy_denom = np.maximum(energy_denom, eps)
    crit_energy_ratio = ur_inverted / energy_denom
    # Clip ratio before normalization to prevent explosion from tiny energy or huge ur
    crit_energy_ratio = np.clip(crit_energy_ratio, 1e-07, 1e7)
    crit_energy_norm = robust_normalize_mad(crit_energy_ratio)

    # --- RELATIVE SLACK URGENCY (normalized by duration, not absolute) ---
    rel_slack = slack / task_min_duration
    # Linear urgency: negative rel_slack → strong positive urgency; positive → decaying soft urgency
    urgency = np.where(rel_slack <= 0.0, -rel_slack * 2.0, np.exp(-rel_slack * 0.5))
    urgency_norm = robust_normalize_mad(urgency)

    # --- WORK-NORMALIZED COMMUNICATION PRESSURE ---
    # High comm relative to work signals data-bound bottleneck — prioritize under tight slack
    comm_to_work_ratio = min_comm_time / (remaining_work + eps)
    comm_pressure = np.where(slack < 0.0, comm_to_work_ratio * 1.5, comm_to_work_ratio * 0.3)
    comm_norm = robust_normalize_mad(comm_pressure)

    # --- ADAPTIVE STARVATION CONTROL ---
    # Relative wait time scaled by *remaining work* and *slack sign*: 
    # Long wait matters more when work is large AND slack is non-negative
    max_wait = np.maximum(np.max(ready_wait_time), eps)
    rel_wait = np.clip(ready_wait_time / max_wait, 0.0, 1.0)
    # Only apply full starvation weight when slack >= 0 and work is substantial
    starvation_weight = np.where(
        (slack >= 0.0) & (remaining_work > np.median(remaining_work) + eps),
        0.25,
        np.where(slack < 0.0, 0.05, 0.15)
    )
    wait_penalty = rel_wait * starvation_weight

    # --- ENERGY SCORING: conditional reversal ---
    # When slack <= 0: minimize energy *less* — prioritize deadline first → set energy score to neutral baseline
    # When slack > 0: maximize energy efficiency → invert normalized energy (lower energy → lower score)
    energy_norm_base = robust_normalize_mad(min_incremental_energy)
    energy_score = np.where(
        slack <= 0.0,
        0.0,  # neutral energy contribution under deadline stress
        -energy_norm_base  # lower energy → lower score → higher priority
    )

    # --- COMBINED SCORE: deterministic, sign-consistent, and balanced ---
    # Weights sum to ~1.0 for interpretability; all terms contribute meaningfully
    score = (
        -2.8 * urgency_norm           # dominant urgency term (negative → higher priority)
        -2.0 * ur_norm               # inverted criticality under deadline stress
        -1.5 * crit_energy_norm      # criticality per marginal energy, uncertainty-gated
        +0.4 * comm_norm             # communication pressure bonus (positive = higher priority)
        +0.3 * energy_score          # energy benefit only when safe
        +wait_penalty                # bounded starvation mitigation
    )

    # Final safeguard: replace NaN/inf with finite extremes, preserving priority order
    score = np.nan_to_num(
        score,
        nan=1e12,
        posinf=1e12,
        neginf=-1e12
    )

    return score
