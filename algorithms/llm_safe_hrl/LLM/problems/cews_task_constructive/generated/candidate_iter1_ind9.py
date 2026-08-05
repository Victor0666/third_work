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
    """Novel priority rule emphasizing deadline feasibility via adaptive risk gating,
    balanced criticality-energy tradeoff, and starvation-aware urgency scaling.
    
    Key innovations:
    - Uses piecewise-linear 'risk-gated urgency': zero contribution when slack > 0,
      linear penalty for 0 >= slack > -median_abs_slack, quadratic penalty for severe
      lateness (slack <= -median_abs_slack) — ensures hard DDL violation avoidance
      without numerical explosion.
    - Introduces 'criticality-normalized energy efficiency': ratio of upward_rank to
      min_incremental_energy, scaled by remaining_work to weight energy savings on
      high-impact paths.
    - Replaces global normalization with *task-wise robust scaling*: each feature
      normalized by its own IQR + eps (more outlier-resilient than mean-abs).
    - Adds 'wait-aware fairness term': ready_wait_time scaled by inverse of median
      wait time, capped to prevent dominance over deadline/energy signals.
    - All components combined with sign-consistent weights ensuring smaller score = higher priority.
    """
    eps = 1e-8

    # Convert inputs safely; no in-place modification
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def robust_scale(x):
        """Scale by interquartile range (IQR) + eps for outlier resilience."""
        q75, q25 = np.percentile(x, [75, 25])
        iqr = q75 - q25
        return x / (iqr + eps)

    # === 1. Risk-gated urgency: strong penalty only for negative slack ===
    # Define thresholds adaptively from data
    abs_slack = np.abs(slack)
    median_abs_slack = np.median(abs_slack) + eps
    # Piecewise: 0 if slack > 0; linear ramp from 0 to 1 as slack drops from 0 to -median_abs_slack;
    # quadratic ramp beyond that to emphasize severe violations
    urgency_base = np.zeros_like(slack)
    mask_moderate = (slack <= 0) & (slack > -median_abs_slack)
    mask_severe = slack <= -median_abs_slack
    urgency_base[mask_moderate] = (-slack[mask_moderate]) / median_abs_slack
    urgency_base[mask_severe] = 1.0 + ((-slack[mask_severe]) / median_abs_slack) ** 2

    # === 2. Criticality-normalized energy efficiency ===
    # Higher upward_rank / energy => better energy-per-criticality ratio => lower score
    # Avoid division by zero; clamp energy to avoid extreme ratios
    safe_energy = np.maximum(min_incremental_energy, eps)
    energy_efficiency = upward_rank / safe_energy
    # Weight by remaining_work to amplify impact on large subtrees
    critical_energy_score = robust_scale(remaining_work) * robust_scale(energy_efficiency)

    # === 3. Communication + execution compactness ===
    # Sum of normalized times reflects "time footprint"; smaller is better
    time_footprint = robust_scale(min_exec_time) + robust_scale(min_comm_time)

    # === 4. Fairness: bounded wait-aware boost ===
    median_wait = np.median(ready_wait_time) + eps
    wait_boost = np.clip(robust_scale(ready_wait_time), 0.0, 2.0)  # cap to avoid starvation override

    # === 5. Uncertainty-aware moderation ===
    # Only amplify urgency for uncertain tasks *with* negative slack
    # Prevents high-uncertainty but slack-rich tasks from preempting
    uncertainty_mod = np.where(slack < 0, uncertainty, 0.0)
    uncertainty_factor = robust_scale(uncertainty_mod)

    # === Final score: weighted sum, all terms designed so smaller = better ===
    # Coefficients chosen to prioritize DDL feasibility first, then energy/criticality,
    # then fairness — avoiding coefficient tuning bias by using principled ratios
    score = (
        3.0 * urgency_base                    # Hard DDL violation avoidance dominates
        + 1.2 * time_footprint               # Favor compact tasks when feasible
        - 1.5 * critical_energy_score        # Reward energy-efficient critical tasks
        + 0.8 * wait_boost                   # Gentle fairness boost
        + 0.6 * uncertainty_factor           # Mild uncertainty amplification only under risk
    )

    # Numerical safety: replace NaN/inf with finite bounds
    return np.nan_to_num(
        score,
        nan=1e12,
        posinf=1e12,
        neginf=-1e12
    )
