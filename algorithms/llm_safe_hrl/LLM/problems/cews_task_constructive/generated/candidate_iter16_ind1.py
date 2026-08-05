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
      - Hard deadline gating with *adaptive urgency cliff* (not just binary)
      - *Risk-scaled criticality*: upward_rank weighted by uncertainty-aware slack penalty
      - *Energy fairness*: min_incremental_energy normalized by task's relative work contribution
      - *Uncertainty-aware waiting boost*: starvation mitigation only when slack > 0 and uncertainty is low
      - *Robust multi-scale fusion*: uses trimmed-mean centering + MAD scaling (more stable than IQR for small N)
      - All terms bounded, eps-protected, and sign-consistent with "lower score = higher priority"
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

    # Robust normalization: trimmed mean center + MAD scale (more stable for N < 5)
    def robust_normalize_mad(x):
        center = np.mean(x) if N <= 3 else np.percentile(x, 50, method='midpoint')
        abs_devs = np.abs(x - center)
        mad = np.median(abs_devs) + eps
        normed = (x - center) / mad
        return np.clip(normed, -8.0, 8.0)

    # === 1. HARD DEADLINE GATING WITH ADAPTIVE URGENCY CLIFF ===
    # Instead of binary mask: smooth transition from safe (slack > 0) to critical (slack << 0)
    # Use tanh-based urgency: steep near zero, bounded, monotonic, avoids division instability
    urgency_base = -slack / (np.maximum(min_exec_time + min_comm_time, eps))
    urgency_cliff = np.tanh(urgency_base * 0.8)  # saturates at ~±0.997 for |urgency_base| > 4
    # Map to [0.0, 2.0]: higher = more urgent; ensure negative slack gives strong pull
    deadline_urgency = 1.0 + 0.95 * np.maximum(0.0, urgency_cliff) + 0.05 * np.where(slack <= 0, 1.0, 0.0)

    # === 2. RISK-SCALED CRITICALITY ===
    # Criticality penalized by both slack risk AND uncertainty: lower priority if high uncertainty *and* tight slack
    risk_factor = np.clip(1.0 + uncertainty * np.maximum(0.0, -slack), 1.0, 5.0)
    crit_risk_weighted = upward_rank / (risk_factor * (min_incremental_energy + eps))
    crit_risk_weighted = np.clip(crit_risk_weighted, 1e-6, 1e6)
    crit_norm = robust_normalize_mad(crit_risk_weighted)

    # === 3. ENERGY FAIRNESS VIA WORK-RELATIVE SCALING ===
    # Energy cost matters more per unit work when remaining_work is small (fine-grained tasks)
    # Normalize energy by work density: energy / (work + eps), then normalize across tasks
    work_density = np.maximum(remaining_work, eps)
    energy_per_work = min_incremental_energy / work_density
    energy_per_work = np.clip(energy_per_work, eps, 1e9)
    energy_norm = robust_normalize_mad(energy_per_work)

    # === 4. UNCERTAINTY-AWARE WAITING BOOST (anti-starvation) ===
    # Only activate waiting boost when task is *not* critically late AND uncertainty is low
    # Prevents starving low-risk, long-wait tasks while avoiding boosting high-risk ones
    wait_eligible = (slack > 0.0) & (uncertainty < np.quantile(uncertainty, 0.5, method='midpoint') + eps)
    max_wait_safe = np.maximum(np.max(ready_wait_time[wait_eligible]) if np.any(wait_eligible) else eps, eps)
    wait_boost_raw = np.where(wait_eligible, ready_wait_time / max_wait_safe, 0.0)
    wait_penalty = np.clip(wait_boost_raw, 0.0, 0.3)  # capped soft boost

    # === 5. ROBUST TIME COST TERM (non-linear, sqrt-smoothed) ===
    time_cost = np.sqrt(np.maximum(min_exec_time, eps)) + np.sqrt(np.maximum(min_comm_time, eps))
    time_norm = robust_normalize_mad(time_cost)

    # === 6. UNCERTAINTY & WORK NORMALIZATION ===
    unc_norm = robust_normalize_mad(uncertainty)
    work_norm = robust_normalize_mad(remaining_work)

    # === FINAL SCORE: linear combination with priority-aligned signs ===
    # Lower score = higher priority → subtract urgency & crit_norm (both good), add others (penalties)
    score = (
        -3.5 * deadline_urgency       # strongest pull: urgency dominates
        - 1.8 * crit_norm             # reward criticality-energy efficiency
        + 0.4 * time_norm             # mild penalty for long-duration tasks (if not urgent)
        + 0.3 * energy_norm           # penalty for high energy-per-work
        + 0.15 * work_norm            # slight penalty for large remaining work (deferred parallelism)
        + 0.2 * unc_norm              # penalty for high uncertainty (favors predictable tasks)
        - wait_penalty                # *boost* for eligible waiting tasks (so lower score)
    )

    # Final guard: replace NaN/inf with finite fallbacks, preserve shape
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    return score
