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
                      normalized waiting fairness, and robust fuzzy slack scaling.
    
    Key mutations vs v1:
      - Replaces soft saturation urgency with hard-zero slack barrier + exponential pressure decay
      - Introduces *slack-aware energy discounting*: only applies energy optimization when slack > median_slack
      - Adds *uncertainty-gated criticality amplification*: upward_rank weighted by uncertainty only if slack > 0
      - Uses *relative waiting time* normalized by workflow-level duration estimate instead of per-task duration
      - Removes comm/comp ratio penalty; replaces with *normalized incremental energy density per MI*
      - All norms now use 5–95 percentile (more stable than 2–98 for small N) and explicit epsilon-guarded division
      - Final score enforces strict priority ordering: urgent tasks (slack <= 0) always get score = -inf (minimally prioritized)
      - Uses np.where-based conditional blending instead of boolean masks to avoid gradient discontinuities in ranking
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

    # Total estimated task duration (execution + communication)
    duration = min_exec_time + min_comm_time + eps

    # Global reference scale: median duration across ready tasks for relative wait normalization
    median_duration = np.median(duration) if N > 1 else duration[0]

    # === URGENT PRIORITY BARRIER (hard deadline enforcement) ===
    # Tasks with slack <= 0 receive ultra-low priority score (-inf), ensuring immediate selection
    # This satisfies hard DDL constraint before any energy/criticality tradeoff
    is_urgent = slack <= 0.0
    urgency_barrier = np.where(is_urgent, -np.inf, 0.0)

    # === SMOOTH URGENCY PRESSURE (for slack > 0 only) ===
    # Exponential decay: higher priority as slack shrinks toward zero — continuous & monotonic
    rel_slack_pos = np.where(slack > 0.0, slack / (duration + eps), 0.0)
    urgency_pressure = np.exp(-rel_slack_pos / (np.median(rel_slack_pos[rel_slack_pos > 0] + eps) + eps))
    # Clamp to avoid underflow artifacts
    urgency_pressure = np.clip(urgency_pressure, eps, 1.0)
    norm_urgency = robust_minmax_norm(urgency_pressure)

    # === RISK-ADJUSTED ENERGY TERM ===
    # Energy density: marginal energy per unit work-time (joules/sec), normalized
    energy_density = min_incremental_energy / (duration + eps)
    energy_density = np.nan_to_num(energy_density, nan=0.0, posinf=0.0, neginf=0.0)
    norm_energy = robust_minmax_norm(energy_density)

    # Slack-aware gating: only optimize energy when slack > median_slack (not just > 0)
    median_slack = np.median(slack) if N > 1 else slack[0]
    energy_opt_mask = (slack > median_slack) & (median_slack > 0.0)
    # Apply discount factor: larger slack → smaller energy weight
    slack_discount = np.where(energy_opt_mask, 1.0 - (slack - median_slack) / (np.max(slack - median_slack + eps) + eps), 0.0)
    norm_energy_weighted = norm_energy * (0.3 + 0.7 * slack_discount)

    # === CRITICALITY AMPLIFICATION ===
    # Upward rank amplified only when slack > 0 AND uncertainty is high → focuses critical path under risk
    norm_upward_rank = robust_minmax_norm(upward_rank)
    unc_median = np.median(uncertainty) if N > 1 else uncertainty[0]
    criticality_amplifier = np.where((slack > 0.0) & (uncertainty > unc_median),
                                   1.0 + 0.5 * robust_minmax_norm(uncertainty),
                                   1.0)
    norm_upward_rank_amp = norm_upward_rank * criticality_amplifier

    # === FAIRNESS BOOST (anti-starvation) ===
    # Relative wait time normalized by global median duration (more stable than per-task duration)
    rel_wait = ready_wait_time / (median_duration + eps)
    rel_wait = np.nan_to_num(rel_wait, nan=0.0, posinf=0.0, neginf=0.0)
    norm_rel_wait = robust_minmax_norm(rel_wait)
    # Boost only for long-waiting tasks (top 30%) regardless of uncertainty
    wait_boost_mask = rel_wait >= np.percentile(rel_wait, 70.0, method='higher') if N > 1 else True
    fairness_boost = np.where(wait_boost_mask, norm_rel_wait * 0.4, 0.0)

    # === WORK-INTENSITY TERM ===
    # Energy per MI (joules/MI) — captures computational efficiency at workload granularity
    energy_per_mi = np.divide(min_incremental_energy, remaining_work + eps,
                              out=np.zeros_like(min_incremental_energy), where=remaining_work + eps != 0)
    energy_per_mi = np.nan_to_num(energy_per_mi, nan=0.0, posinf=0.0, neginf=0.0)
    norm_energy_per_mi = robust_minmax_norm(energy_per_mi)

    # === COMBINED SCORE (smaller = better) ===
    # Base components weighted to preserve urgency dominance, then add corrections
    score_base = (
        0.45 * norm_urgency +
        0.25 * norm_energy_weighted +
        0.15 * norm_upward_rank_amp +
        0.08 * norm_energy_per_mi +
        0.07 * (1.0 - fairness_boost)
    )

    # Apply urgency barrier: overwrite score with -inf for all urgent tasks
    score = np.where(is_urgent, -np.inf, score_base)

    # Hard bounds and NaN/inf cleanup
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)

    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
