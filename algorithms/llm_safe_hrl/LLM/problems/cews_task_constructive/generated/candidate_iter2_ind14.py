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
    Priority rule combining robust deadline gating (sigmoid), critical-path-aware energy efficiency,
    starvation-avoiding aging, and uncertainty-aware urgency — all normalized via IQR.
    
    Key improvements:
    - Uses sigmoid deadline risk (bounded, smooth) but adds hard penalty floor for slack < -1e-3
    - Energy efficiency: min_incremental_energy / (remaining_work + eps) → prioritizes energy savings on high-work critical paths
    - Upward rank modulates energy term only above median (avoids trivial node over-prioritization)
    - Ready-wait boost activates only after percentile-based threshold (prevents premature aging)
    - Uncertainty urgency gated by slack: only scheduled early if slack permits; otherwise deferred
    - All features normalized with robust IQR scaling; final score clipped to finite bounds
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

    def robust_normalize(x):
        q1, q3 = np.quantile(x, [0.25, 0.75])
        iqr = q3 - q1
        scale = iqr if iqr > eps else np.mean(np.abs(x)) + eps
        return x / (scale + eps)

    # Deadline risk: smooth sigmoid + hard floor for severely negative slack
    slack_abs_med = np.abs(np.median(slack)) + eps
    sigmoid_risk = 1.0 / (1.0 + np.exp(-slack / slack_abs_med))
    hard_penalty = np.where(slack < -1e-3, -slack * 10.0, 0.0)  # Strong linear penalty for actual violation risk
    deadline_risk = sigmoid_risk + hard_penalty

    # Latency pressure: execution + communication time
    latency_pressure = min_exec_time + min_comm_time
    norm_latency = robust_normalize(latency_pressure)

    # Energy efficiency: marginal energy per unit remaining work (joules/MI)
    eff_ratio = min_incremental_energy / (remaining_work + eps)
    norm_eff_ratio = robust_normalize(eff_ratio)

    # Criticality modulation: only apply upward rank weight where it matters
    ur_median = np.median(upward_rank)
    critical_upward = np.where(upward_rank >= ur_median, upward_rank, 0.0)
    norm_upward = robust_normalize(critical_upward)

    # Starvation avoidance: wait boost only beyond 90th percentile of current waits
    wait_thresh = np.percentile(ready_wait_time, 90) if len(ready_wait_time) > 1 else np.max(ready_wait_time) + eps
    waiting_boost = np.where(ready_wait_time > wait_thresh, ready_wait_time, 0.0)
    norm_waiting = robust_normalize(waiting_boost)

    # Uncertainty urgency: only activated when slack is non-critical (avoids worsening deadlines)
    slack_safe = np.where(slack >= 0, slack, 0.0)  # ignore uncertainty when already late
    uncertainty_urgency = uncertainty * (1.0 + robust_normalize(slack_safe))
    norm_uncertainty = robust_normalize(uncertainty_urgency)

    # Final score: lower = better
    # Weights tuned to emphasize deadline safety first, then energy efficiency and starvation
    score = (
        -4.0 * robust_normalize(deadline_risk)      # Strongest weight: enforce DDL
        + 0.25 * norm_latency                       # Prefer low-latency tasks when safe
        + 0.3 * norm_eff_ratio                      # Favor energy-efficient high-work tasks
        + 0.1 * norm_uncertainty                    # Early-schedule uncertain-but-safe tasks
        + 0.15 * norm_waiting                       # Boost long-waiting tasks moderately
        - 0.2 * norm_upward                         # Prioritize critical path *only* where beneficial
    )

    # Ensure finite output, no NaN/inf
    return np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
