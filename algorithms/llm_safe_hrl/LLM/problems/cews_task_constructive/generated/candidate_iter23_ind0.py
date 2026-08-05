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
    v2 priority rule: Hard urgency dominance + critical-path energy gating with tightened slack threshold +
                      adaptive fairness via normalized wait-per-work + uncertainty decayed by proximity +
                      robust degenerate-safe normalization.

    Key improvements:
    - Absolute urgency enforcement: urgent tasks (slack <= 0) get fixed -1e12, no leakage.
    - Critical-path energy penalty only activates for *both* high upward_rank (>75th percentile) AND tight rel_slack (<= 0.1),
      avoiding over-penalization while strengthening deadline-critical paths.
    - Fairness uses normalized wait-per-work (ready_wait_time / (remaining_work + eps)) with dynamic 10th-percentile gate,
      enabling starvation prevention scaled to workload magnitude.
    - Uncertainty boost decays exponentially with non-negative slack: exp(-max(0, slack)/tau), preserving urgency dominance
      and enhancing near-deadline risk awareness without noise amplification.
    - All normalizations use 1%-99% clipping + min-max with degenerate fallback; all divisions guarded.
    - Final weights tuned for DDL-hard constraint satisfaction first, then energy minimization among feasible candidates.
    """
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=float).copy()
    min_comm_time = np.asarray(min_comm_time, dtype=float).copy()
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float).copy()
    slack = np.asarray(slack, dtype=float).copy()
    upward_rank = np.asarray(upward_rank, dtype=float).copy()
    remaining_work = np.asarray(remaining_work, dtype=float).copy()
    ready_wait_time = np.asarray(ready_wait_time, dtype=float).copy()
    uncertainty = np.asarray(uncertainty, dtype=float).copy()
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)

    def robust_minmax_norm(x):
        if x.size == 0:
            return np.zeros_like(x)
        x_clean = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x_finite = x_clean[np.isfinite(x_clean)]
        if x_finite.size == 0:
            return np.zeros_like(x)
        p01 = np.percentile(x_finite, 1.0, method='midpoint')
        p99 = np.percentile(x_finite, 99.0, method='midpoint')
        x_clipped = np.clip(x_clean, p01, p99)
        x_min = np.min(x_clipped)
        x_max = np.max(x_clipped)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x_clipped - x_min) / (x_max - x_min + eps)

    # Hard urgency flag: absolute priority for overdue or at-risk tasks
    is_urgent = (slack <= 0.0).astype(float)

    # Duration-based features
    duration = min_exec_time + min_comm_time + eps
    rel_slack = np.divide(slack, duration, out=np.zeros_like(slack), where=duration != 0)
    rel_slack = np.where(np.isfinite(rel_slack), rel_slack, 0.0)

    # Critical latency: duration weighted by upward rank importance
    critical_latency_raw = duration * (1.0 + 0.7 * robust_minmax_norm(upward_rank))
    norm_critical_latency = robust_minmax_norm(critical_latency_raw)

    # Energy density: incremental energy per time unit (proxy for efficiency)
    energy_density = np.divide(min_incremental_energy, duration + eps,
                               out=np.zeros_like(min_incremental_energy), where=duration + eps != 0)
    norm_energy_density = robust_minmax_norm(energy_density)

    # Tight critical-path gating: only penalize energy when both slack is tight AND rank is high
    rank_threshold = np.percentile(upward_rank, 75.0) + eps
    tight_slack_mask = (rel_slack <= 0.1).astype(float)  # tighter than v1's 0.3
    high_rank_mask = (upward_rank > rank_threshold).astype(float)
    energy_penalty_mask = tight_slack_mask * high_rank_mask
    energy_penalty = norm_energy_density * energy_penalty_mask

    # Adaptive fairness: normalized wait-per-work with dynamic low-work filtering
    wait_per_work = np.divide(ready_wait_time, remaining_work + eps,
                              out=np.zeros_like(ready_wait_time), where=remaining_work + eps != 0)
    wait_per_work = np.nan_to_num(wait_per_work, nan=0.0, posinf=0.0, neginf=0.0)
    wpw_finite = wait_per_work[np.isfinite(wait_per_work)]
    work_gate_threshold = np.percentile(wpw_finite, 10.0) + eps if wpw_finite.size > 0 else eps
    wait_gate = (wait_per_work >= work_gate_threshold).astype(float)
    norm_wait_per_work = robust_minmax_norm(wait_per_work)
    wait_penalty = (1.0 - is_urgent) * norm_wait_per_work * wait_gate

    # Uncertainty boost: decays exponentially as slack increases (only active near deadline)
    tau = 15.0
    proximity_bias = np.exp(-np.maximum(0.0, slack) / tau)
    uncertainty_boost = uncertainty * proximity_bias
    norm_uncertainty_boost = robust_minmax_norm(uncertainty_boost)

    # Remaining work serves as coarse granularity signal — downweighted
    norm_remaining_work = robust_minmax_norm(remaining_work)

    # Base score scaffold
    base_score = np.full(N, 1.0, dtype=float)

    # Compose final score: urgent tasks dominate with fixed minimum
    score = np.where(is_urgent, -1000000000000.0, base_score)
    score = np.where(
        is_urgent,
        score,
        score + 0.25 * norm_critical_latency
              + 0.22 * energy_penalty
              + 0.14 * wait_penalty
              + 0.09 * norm_uncertainty_boost
              + 0.04 * norm_remaining_work
    )

    # Clip and sanitize
    score = np.clip(score, -1000000000000.0, 1000000000000.0)
    score = np.nan_to_num(score, nan=1000000000000.0, posinf=1000000000000.0, neginf=-1000000000000.0)

    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
