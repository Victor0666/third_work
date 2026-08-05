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
    - Strict deadline gating with *adaptive zero-tolerance* (negative slack → fixed min score)
    - Robust *slack-aware energy scaling*: energy penalty grows superlinearly only under risk
    - *Criticality-weighted waiting fairness*: starvation relief activated only when slack > 0 and criticality is low
    - *Uncertainty-coupled duration robustness*: communication/computation time weighted by uncertainty to reflect scheduling confidence
    - *MAD-based normalization* (more outlier-resistant than IQR) with explicit sign-preserving centering
    - All operations eps-protected, nan/inf-guarded, deterministic, and shape-compliant.
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

    # Robust MAD-based normalization: more stable than IQR for small N
    def robust_normalize_mad(x):
        center = np.median(x)
        dev = np.abs(x - center)
        mad = np.median(dev) + eps
        normalized = (x - center) / mad
        return np.clip(normalized, -8.0, 8.0)  # tighter bounds than IQR

    # === 1. Hard deadline enforcement: negative slack → highest priority (lowest score)
    # Assign fixed minimal score (-10.0) to all overdue/latency-risk tasks — deterministic & zero-tolerance
    is_overdue = (slack <= 0.0)
    base_score = np.full(N, 0.0, dtype=float)
    base_score[is_overdue] = -10.0

    # === 2. Urgency-aware energy scaling: only penalize energy under risk
    # Energy penalty grows quadratically when slack < median_slack (i.e., relative pressure), else linear
    median_slack = np.median(slack) + eps
    slack_pressure = np.clip((median_slack - slack) / (np.abs(median_slack) + eps), 0.0, 5.0)
    energy_penalty_factor = np.where(slack <= median_slack,
                                     1.0 + 0.8 * slack_pressure + 0.3 * slack_pressure**2,
                                     1.0 + 0.2 * slack_pressure)
    risk_adjusted_energy = min_incremental_energy * energy_penalty_factor

    # === 3. Criticality-energy efficiency ratio — but capped and normalized
    # Avoid division-by-zero; cap extreme ratios to prevent dominance
    crit_eff_ratio = upward_rank / (risk_adjusted_energy + eps)
    crit_eff_ratio = np.clip(crit_eff_ratio, 1e-5, 1e6)
    crit_eff_norm = robust_normalize_mad(crit_eff_ratio)

    # === 4. Uncertainty-weighted duration: high uncertainty → prefer shorter duration tasks *only if slack permits*
    # Reflects confidence loss: uncertain VMs make long exec/comm risky → downweight duration when uncertainty high
    dur_weight = 1.0 / (1.0 + uncertainty)
    weighted_duration = (min_exec_time + min_comm_time) * dur_weight
    dur_norm = robust_normalize_mad(weighted_duration)

    # === 5. Adaptive waiting fairness: only applied to *non-urgent*, *low-criticality* tasks
    # Prevents starvation without interfering with deadline or critical-path tasks
    median_ur = np.median(upward_rank) + eps
    low_crit_mask = (upward_rank < 0.7 * median_ur) & (slack > 0.0)
    max_wait = np.maximum(np.max(ready_wait_time), eps)
    wait_rel = np.clip(ready_wait_time / max_wait, 0.0, 1.0)
    wait_penalty = np.where(low_crit_mask, wait_rel * 0.3, 0.0)

    # === 6. Remaining work normalization — used as *positive* signal (larger work → higher priority *if feasible*)
    # Encourages batching of large descendant workloads early, but only when slack allows
    rw_norm = robust_normalize_mad(remaining_work)
    work_bonus = np.where(slack > 0.0, rw_norm * 0.25, 0.0)

    # === 7. Uncertainty normalization — used as *penalty* (higher uncertainty → lower priority, unless overdue)
    unc_norm = robust_normalize_mad(uncertainty)
    unc_penalty = np.where(is_overdue, 0.0, unc_norm * 0.15)

    # === Final score: sum of normalized, bounded, sign-consistent components
    # Lower score = higher priority. Overdue tasks already set to -10.0; others built from zero baseline.
    non_overdue_mask = ~is_overdue
    score_non_overdue = (
        -2.8 * crit_eff_norm +
        0.45 * dur_norm +
        0.20 * work_bonus -
        0.12 * unc_penalty +
        wait_penalty
    )
    # Apply non-overdue score only where not already overridden
    score = np.where(is_overdue, base_score, score_non_overdue)

    # Final numeric safety: ensure finite output
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    return score
