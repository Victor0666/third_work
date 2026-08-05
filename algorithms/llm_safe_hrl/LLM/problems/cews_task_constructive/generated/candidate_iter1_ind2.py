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
    """Novel priority rule emphasizing deadline feasibility first, then energy-aware criticality,
    with starvation-avoidance and uncertainty-calibrated urgency.
    
    Key innovations:
      - Uses *slack-based urgency gating*: only tasks with slack < 0 or slack < median_slack
        activate high-priority correction terms (avoids over-penalizing distant tasks).
      - Replaces linear normalization with *robust range scaling* (IQR-based) for better
        outlier resistance and physical interpretability.
      - Introduces *criticality-energy ratio*: upward_rank / (min_exec_time + min_comm_time + eps)
        to favor high-impact tasks per unit time-cost — prioritizes throughput-critical work.
      - Models *uncertainty-adjusted waiting incentive*: ready_wait_time scaled by (1 + uncertainty)
        to gently boost long-waiting tasks only when risk is non-negligible.
      - Applies *sign-consistent penalty stacking*: negative slack contributes a large *additive*
        penalty (not multiplicative), ensuring hard DDL compliance dominates; all other terms
        are normalized and weighted to preserve comparability.
    """
    eps = 1e-8

    # Safe array conversion without mutation
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    # Robust scaling: Interquartile Range + mean absolute deviation fallback
    def robust_scale(x):
        q1, q3 = np.quantile(x, [0.25, 0.75])
        iqr = q3 - q1
        if iqr < eps:
            # Fallback to mean abs deviation if IQR near zero
            mad = np.mean(np.abs(x - np.median(x)))
            scale = mad if mad > eps else np.mean(np.abs(x)) + eps
        else:
            scale = iqr + eps
        return x / scale

    # --- Deadline Risk Priority (Hard constraint enforcement) ---
    # Strong additive penalty for negative slack; moderate penalty for low-slack tasks
    # Gate: only apply full penalty to tasks in top 30% of urgency (slack <= 30th percentile)
    slack_sorted = np.sort(slack)
    slack_thresh = slack_sorted[max(0, int(0.3 * len(slack_sorted)))] if len(slack_sorted) > 0 else 0.0
    is_urgent = (slack <= slack_thresh)
    deadline_penalty = np.where(is_urgent, 
                               np.maximum(-slack, 0.0) * 3.0 + np.where(slack < 0, 5.0, 0.0),
                               0.0)

    # --- Criticality-Energy Efficiency Ratio ---
    # Favor tasks with high upward_rank per unit execution+comm cost
    time_cost = min_exec_time + min_comm_time + eps
    crit_eff_ratio = upward_rank / time_cost  # higher = more critical per time unit
    # Normalize: invert to make high ratio → low score (higher priority)
    crit_eff_score = -robust_scale(crit_eff_ratio)

    # --- Energy & Resource Cost Terms (normalized, positive weights) ---
    exec_score = robust_scale(min_exec_time)
    comm_score = robust_scale(min_comm_time)
    energy_score = robust_scale(min_incremental_energy)

    # --- Starvation Avoidance & Uncertainty Modulation ---
    # Boost wait time only where uncertainty > 0.1 (non-trivial risk)
    wait_boost = np.where(uncertainty > 0.1, 
                         robust_scale(ready_wait_time) * (1.0 + uncertainty), 
                         robust_scale(ready_wait_time) * 0.3)
    # Invert: longer wait → lower score → higher priority
    wait_score = -wait_boost

    # --- Uncertainty-Aware Criticality Dampening ---
    # Reduce priority of high-uncertainty tasks *unless* they're urgent
    # Prevents risky tasks from dominating unless deadline-constrained
    unc_damp = np.where(is_urgent, 1.0, 1.0 - np.clip(uncertainty, 0.0, 0.8))
    crit_eff_score = crit_eff_score * unc_damp

    # --- Final weighted combination ---
    # All components are now on comparable scale (-1 to +1 approx); sum with tuned weights
    score = (
        0.30 * exec_score +
        0.20 * comm_score +
        0.25 * energy_score +
        1.00 * deadline_penalty +  # Dominant term for hard DDL
        0.15 * crit_eff_score +
        0.10 * wait_score
    )

    # Numerical safety: clamp extremes, replace NaN/inf
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)

    return score
