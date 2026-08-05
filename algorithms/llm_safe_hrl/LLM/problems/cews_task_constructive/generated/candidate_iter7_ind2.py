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
    - Slack-gated urgency via adaptive sigmoidal gating (not binary mask)
    - Risk-adjusted energy density scaled by *normalized remaining work* and *uncertainty-aware denominator*
    - Bounded, min-max normalized features (not IQR) for cross-seed stability & outlier resilience
    - Starvation control via *relative wait time* (wait / max(1, task_min_duration)) capped and slack-modulated
    - Criticality-energy tradeoff using *upward_rank × (1 + |slack|/max_slack) / (energy × (1 + uncertainty))*
    - Explicit penalty for high uncertainty only when slack is tight (|slack| < median_duration)
    - All operations epsilon-protected, nan/inf-cleaned, deterministic, shape-compliant.
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

    # Robust min-max normalization with clipping to prevent degeneracy
    def robust_minmax(x):
        x_min, x_max = np.min(x), np.max(x)
        range_val = x_max - x_min + eps
        normalized = (x - x_min) / range_val
        return np.clip(normalized, 0.0, 1.0)

    # Task intrinsic duration and derived metrics
    task_min_duration = min_exec_time + min_comm_time + eps
    max_slack = np.maximum(np.abs(np.median(slack)), np.max(np.abs(slack))) + eps
    rel_slack_abs = np.abs(slack) / max_slack  # normalized lateness pressure

    # === 1. Slack-gated urgency: smooth sigmoidal gate, no hard threshold ===
    # Stronger penalty for negative slack; decays smoothly beyond 2× median |slack|
    urgency_gate = 1.0 / (1.0 + np.exp(-4.0 * (rel_slack_abs - 0.5)))  # centered at 0.5 → ~0.5 at median slack
    urgency_penalty = np.where(slack < 0, 1.0 + 2.0 * (-slack / (task_min_duration + eps)), urgency_gate)

    # === 2. Risk-aware energy density: work-contextualized & uncertainty-dampened ===
    # Scale energy by remaining work *only if work > median*, else neutral
    median_rw = np.median(remaining_work) + eps
    rw_scale = np.where(remaining_work > median_rw, 
                        np.clip(remaining_work / median_rw, 1.0, 5.0), 
                        1.0)
    # Denominator: energy × (1 + uncertainty), but avoid zero or explosion
    energy_denom = np.maximum(min_incremental_energy * (1.0 + uncertainty), eps)
    # Criticality-energy efficiency: higher rank / lower risk-adjusted energy
    crit_eff = upward_rank / energy_denom * rw_scale
    crit_eff_norm = robust_minmax(crit_eff)

    # === 3. Starvation control: relative wait, bounded & slack-modulated ===
    # Wait relative to task's own duration, capped at 1.0, then scaled down under deadline pressure
    rel_wait = np.clip(ready_wait_time / task_min_duration, 0.0, 1.0)
    # Reduce starvation boost when slack is very tight (urgent tasks shouldn't be penalized for waiting)
    wait_boost = rel_wait * np.clip(1.0 - rel_slack_abs, 0.0, 1.0) * 0.3

    # === 4. Uncertainty penalty: active only when slack is tight (|slack| < median_duration) ===
    median_dur = np.median(task_min_duration) + eps
    unc_penalty = np.where(np.abs(slack) < median_dur,
                           uncertainty * 0.2,
                           0.0)

    # === 5. Time cost penalty: sqrt-normalized duration, normalized ===
    time_cost = np.sqrt(np.maximum(min_exec_time, eps)) + np.sqrt(np.maximum(min_comm_time, eps))
    time_norm = robust_minmax(time_cost)

    # === 6. Work importance: normalized remaining work, linearly weighted ===
    work_norm = robust_minmax(remaining_work)

    # === Composite score: smaller = better ===
    # Prioritize urgency (high penalty → high score → low priority ⇒ so invert urgency_penalty sign)
    # We want high urgency_penalty → low score ⇒ use -urgency_penalty
    # crit_eff_norm: higher = better → keep positive weight
    # Others: time_norm, unc_penalty, wait_boost → all penalties → positive weights
    score = (
        -2.8 * urgency_penalty       # dominant deadline enforcement
        + 1.5 * crit_eff_norm        # reward critical+efficient tasks
        + 0.4 * time_norm            # mild penalty for long-duration tasks
        + 0.3 * work_norm            # slight boost for high-work subtrees
        + 0.25 * unc_penalty         # targeted uncertainty penalty
        + 0.2 * wait_boost           # bounded fairness term
    )

    # Final cleanup: replace NaN/inf with large finite fallbacks (smaller score = higher priority)
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    
    # Ensure finite & bounded
    score = np.clip(score, -1e8, 1e8)
    
    return score
