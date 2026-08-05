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
    'Enhanced priority rule unifying deadline-hardness, risk-aware energy efficiency, and starvation fairness.\n    Combines Parent 2\'s slack gating and energy-per-second focus with Parent 1\'s bounded urgency and robust IQR scaling.\n    Novel improvements: (1) adaptive slack penalty with capped exponential + linear fallback for stability;\n    (2) uncertainty-gated criticality leveraging both slack and uncertainty thresholds;\n    (3) starvation guard activated only when *both* slack < 0 AND ready_wait_time > median_wait;\n    (4) unified robust normalization with guaranteed fallback to avoid division-by-zero or NaN.\n    All operations are finite, deterministic, and satisfy interface contract.'
    eps = 1e-08
    # Ensure float64 and copy to avoid in-place modification
    min_exec_time = np.asarray(min_exec_time, dtype=float).copy()
    min_comm_time = np.asarray(min_comm_time, dtype=float).copy()
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float).copy()
    slack = np.asarray(slack, dtype=float).copy()
    upward_rank = np.asarray(upward_rank, dtype=float).copy()
    remaining_work = np.asarray(remaining_work, dtype=float).copy()
    ready_wait_time = np.asarray(ready_wait_time, dtype=float).copy()
    uncertainty = np.asarray(uncertainty, dtype=float).copy()

    # Robust normalization with guaranteed fallback: IQR first, then mean-abs if degenerate
    def robust_normalize(x):
        if x.size == 0:
            return np.zeros_like(x)
        q1, q3 = np.quantile(x, [0.25, 0.75], method='midpoint')
        iqr = q3 - q1
        med = np.median(x)
        if iqr < eps:
            # Fallback: normalize by mean absolute deviation from median
            mad = np.mean(np.abs(x - med)) + eps
            return (x - med) / mad
        return (x - med) / (iqr + eps)

    # === Deadline Risk: Bounded & Adaptive Slack Urgency ===
    # Use capped exponential for negative slack (hard violation), linear decay for positive slack
    # Avoids explosion and ensures smooth transition at slack=0
    slack_urgency = np.where(
        slack <= 0,
        np.clip(10.0 + 5.0 * np.abs(slack), 10.0, 1e4),  # Cap max penalty
        np.clip(1.0 / (1.0 + 0.01 * slack), 1e-4, 0.999)  # Smooth decay for safe slack
    )

    # === Energy Efficiency: Power-aware (J/s) with hard caps to prevent outliers ===
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_second = min_incremental_energy / duration
    # Clip extreme values before normalization to preserve ranking stability
    capped_energy_ps = np.clip(energy_per_second, -1e6, 1e6)
    norm_energy_ps = robust_normalize(capped_energy_ps)

    # === Criticality-Energy Coupling: Upward rank scaled by work-normalized energy efficiency ===
    # Leverage remaining_work to down-weight trivial but high-rank nodes; avoid division by zero
    work_efficiency_ratio = remaining_work / (min_incremental_energy + eps)
    # Only activate criticality boost when task is both important *and* energy-efficient
    # Gated by slack violation OR high uncertainty to prioritize under risk
    slack_or_uncert_mask = np.logical_or(slack <= 0, uncertainty > np.percentile(uncertainty, 75, method='midpoint') + eps)
    leveraged_rank = upward_rank * np.where(slack_or_uncert_mask, work_efficiency_ratio, 1.0)
    norm_leveraged_rank = robust_normalize(leveraged_rank)

    # === Uncertainty-Gated Boost: Only amplify priority when slack is tight AND uncertainty is high ===
    # Tight slack: slack <= 30s OR violated (slack <= 0); high uncertainty: > Q75
    tight_slack_mask = np.logical_or(slack <= 0, slack <= 30.0)
    high_uncert_mask = uncertainty > (np.percentile(uncertainty, 75, method='midpoint') + eps)
    uncertainty_boost = np.where(tight_slack_mask & high_uncert_mask, robust_normalize(uncertainty), 0.0)

    # === Starvation Guard: Activate *only* for long-waiting tasks *under deadline pressure* ===
    # Prevents idle starvation without diluting urgency — requires both slack violation and long wait
    median_wait = np.median(ready_wait_time) if ready_wait_time.size > 1 else 0.0
    starvation_guard = np.where(
        (slack <= 0) & (ready_wait_time > median_wait + eps),
        robust_normalize(ready_wait_time),
        0.0
    )

    # === Base score: weighted sum — smaller is better ===
    # Prioritize urgency (negative weight), penalize energy inefficiency (positive weight)
    score = (
        +0.35 * norm_energy_ps           # penalize high J/s
        - 1.8 * slack_urgency            # strongly reward deadline safety
        + 0.15 * norm_leveraged_rank     # reward critical+efficient tasks
        + 0.1 * uncertainty_boost        # nudge uncertain risky tasks forward
        - 0.05 * starvation_guard        # slight boost to starving urgent tasks (fairness)
        + 0.05 * robust_normalize(min_exec_time)      # minor exec time penalty
        + 0.05 * robust_normalize(min_comm_time)      # minor comm time penalty
    )

    # Final safeguard: ensure finite output, no inf/nan
    return np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
