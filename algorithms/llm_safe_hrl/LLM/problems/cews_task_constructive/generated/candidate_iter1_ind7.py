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
      - Uses *slack-gated* weighting: only tasks with slack < 0 get boosted urgency;
        others use normalized latency-energy tradeoff.
      - Introduces 'criticality-adjusted energy' = min_incremental_energy / (upward_rank + eps),
        penalizing high-energy use on critical path nodes.
      - Replaces static normalization with robust quantile-based scaling (IQR + median),
        avoiding sensitivity to outliers and zero-mass distributions.
      - Models waiting time as a *soft floor*: long-waiting tasks get bounded priority boost
        (logistic-shaped), preventing starvation without overwhelming deadline signals.
      - Combines execution + communication into 'latency burden', then normalizes jointly
        with energy to reflect resource-time-energy coupling.
      - All terms are finite, deterministic, and preserve sign semantics for priority ordering.
    """
    eps = 1e-8

    # Convert inputs safely; no in-place mutation
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    # Robust quantile-based normalization: scale by IQR + eps, center at median
    def robust_normalize(x):
        q25, q50, q75 = np.quantile(x, [0.25, 0.5, 0.75])
        iqr = q75 - q25
        denom = iqr + eps
        return (x - q50) / denom

    # 1. Deadline risk: strong penalty for negative slack; zero otherwise
    # Use linear penalty scaled by absolute magnitude, capped at 3x typical slack range
    slack_abs = np.abs(slack)
    slack_range = np.maximum(np.ptp(slack), eps)
    deadline_risk = np.where(slack < 0, -slack / (slack_range + eps), 0.0)

    # 2. Latency burden: sum exec + comm, normalized jointly (not separately)
    latency_burden = min_exec_time + min_comm_time
    norm_latency = robust_normalize(latency_burden)

    # 3. Criticality-adjusted energy: reward low energy *on critical paths*
    # Higher upward_rank => more critical => stricter energy budget per importance unit
    adj_energy = min_incremental_energy / (upward_rank + eps)
    norm_adj_energy = robust_normalize(adj_energy)

    # 4. Starvation mitigation: logistic boost for long-waiting tasks
    # Saturates at ~2x max wait time, avoids unbounded growth
    wait_max = np.max(ready_wait_time) + eps
    wait_ratio = np.clip(ready_wait_time / (2.0 * wait_max + eps), 0.0, 1.0)
    starvation_boost = 1.0 - 1.0 / (1.0 + np.exp(4.0 * (wait_ratio - 0.5)))

    # 5. Uncertainty modulation: only amplify priority for high-uncertainty *and* tight-slack tasks
    # Avoids over-prioritizing uncertain-but-plentiful slack tasks
    uncertainty_mod = uncertainty * np.where(slack < 0.1 * (np.mean(np.abs(slack)) + eps), 1.0, 0.0)
    norm_uncertainty = robust_normalize(uncertainty_mod)

    # 6. Remaining work as coarse load indicator — normalize, but down-weight
    norm_work = robust_normalize(remaining_work) * 0.3

    # Priority score: smaller = better
    # Negative weights for deadline_risk and starvation_boost (they *improve* priority when active)
    # Positive weights for latency, energy, uncertainty (they *hurt* priority)
    score = (
        0.30 * norm_latency
        + 0.35 * norm_adj_energy
        + 0.15 * norm_uncertainty
        + 0.10 * norm_work
        - 3.00 * deadline_risk   # Strong hard-DDL enforcement
        - 0.25 * starvation_boost  # Bounded soft fairness
    )

    # Final numerical guard: ensure finite output, shape (N,)
    return np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
