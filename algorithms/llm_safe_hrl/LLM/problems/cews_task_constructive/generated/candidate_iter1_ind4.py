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
    """Novel priority rule emphasizing deadline-criticality, risk-aware energy efficiency,
    and starvation prevention via adaptive normalization and slack-driven gating.

    Key innovations:
      - Slack gating: only activate uncertainty & waiting-time boosts for tasks with slack <= 0,
        avoiding premature interference when deadlines are safe.
      - Dual-phase normalization: use robust IQR-based scaling for stability under outliers,
        fallback to mean-abs if IQR is near-zero.
      - Energy-efficiency ratio: combine incremental energy with total duration (exec+comm)
        to reward energy-per-second efficiency — penalizing high-energy/low-work tasks.
      - Critical-path leverage: scale upward_rank by normalized remaining_work to avoid
        over-prioritizing tiny but highly-ranked nodes.
      - Starvation guard: ready_wait_time contributes *only* when slack < 0 or upward_rank > median,
        ensuring fairness without undermining urgency.
      - All terms designed to be finite, deterministic, and numerically stable.
    """
    eps = 1e-8

    # Safe conversion without mutation
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    # Robust normalization: IQR-based, fallback to mean-abs
    def robust_normalize(x):
        q1, q3 = np.quantile(x, [0.25, 0.75], method='midpoint')
        iqr = q3 - q1
        if iqr < eps:
            return x / (np.mean(np.abs(x)) + eps)
        return x / (iqr + eps)

    # Deadline urgency: exponential decay for slack > 0, hard penalty for slack <= 0
    # Avoids log(0) and overflow; negative slack mapped to large positive penalty
    slack_urgency = np.where(
        slack <= 0,
        10.0 + np.abs(slack) / (np.maximum(np.mean(np.abs(slack)), eps) + eps),
        np.exp(-slack / (np.maximum(np.mean(np.abs(slack)), eps) + eps))
    )

    # Energy efficiency ratio: marginal energy per unit time (exec + comm), lower is better
    # Prevent division by zero; cap extreme values
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_second = min_incremental_energy / duration
    # Normalize after capping extremes to prevent outlier dominance
    capped_energy_ps = np.clip(energy_per_second, -1e6, 1e6)
    norm_energy_ps = robust_normalize(capped_energy_ps)

    # Critical-path leverage: upward_rank weighted by relative remaining work
    # Prevents over-scoring low-work critical nodes (e.g., tiny merge task on long path)
    work_scale = robust_normalize(remaining_work)
    leveraged_rank = upward_rank * (1.0 + work_scale)

    # Uncertainty boost only when deadline is tight (slack <= 0) — avoids noise in safe region
    uncertainty_boost = np.where(
        slack <= 0,
        robust_normalize(uncertainty),
        0.0
    )

    # Starvation guard: ready_wait_time only activates if either deadline is violated
    # or task lies on high-importance path (upward_rank above median)
    median_rank = np.median(upward_rank) if len(upward_rank) > 1 else 0.0
    wait_activation = np.logical_or(slack <= 0, upward_rank > median_rank)
    norm_wait = np.where(wait_activation, robust_normalize(ready_wait_time), 0.0)

    # Final score: smaller = higher priority
    # Negative weights for urgency, energy-efficiency, and starvation guard → push down score
    # Positive weights for uncertainty boost and leveraged rank → push up score (lower priority)
    score = (
        + 0.30 * norm_energy_ps           # prefer energy-efficient tasks
        - 1.50 * slack_urgency           # strong pull for urgent/negative-slack tasks
        + 0.20 * robust_normalize(leveraged_rank)   # moderate cost for critical-path weight
        + 0.15 * uncertainty_boost       # mild cost for high-risk urgent tasks
        - 0.10 * norm_wait               # gentle pull for starved tasks under pressure
        + 0.10 * robust_normalize(min_exec_time)     # light penalty for long exec (secondary)
        + 0.05 * robust_normalize(min_comm_time)     # light penalty for long comm (tertiary)
    )

    # Final numerical safeguard
    return np.nan_to_num(
        score,
        nan=1e12,
        posinf=1e12,
        neginf=-1e12
    )
