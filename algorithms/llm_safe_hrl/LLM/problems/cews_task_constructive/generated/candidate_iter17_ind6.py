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
    v2 priority rule: Absolute urgency dominance + criticality-gated risk coupling + trimmed robust fairness.
    
    Key innovations:
    - Hard urgency enforcement via -1e12 offset (guarantees urgent tasks always win)
    - Critical-path latency penalty scaled by upward_rank and duration, normalized robustly
    - Energy penalty using upward-rank-normalized density with 95th-percentile thresholding
    - Fairness via trimmed-mean wait-time scaling (avoids outlier distortion) for non-urgent tasks
    - Uncertainty boost gated jointly by |slack| and upward_rank to suppress noise on low-criticality tasks
    - All divisions guarded; NaN/inf sanitized; robust 1%-99% clipping for normalization
    - Deterministic, no side effects, fully vectorized
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
        p01 = np.percentile(x, 1.0, method='midpoint')
        p99 = np.percentile(x, 99.0, method='midpoint')
        x_clipped = np.clip(x, p01, p99)
        x_min = np.min(x_clipped)
        x_max = np.max(x_clipped)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x_clipped - x_min) / (x_max - x_min + eps)

    # Absolute urgency dominance: urgent tasks (slack <= 0) get minimal score
    is_urgent = (slack <= 0).astype(float)
    base_score = np.where(is_urgent, -1e12, 0.0)

    # Critical-path latency penalty: duration weighted by rank importance
    duration = min_exec_time + min_comm_time + eps
    critical_latency_raw = duration * (1.0 + 0.8 * robust_minmax_norm(upward_rank))
    norm_critical_latency = robust_minmax_norm(critical_latency_raw)

    # Energy penalty: rank-normalized energy density, thresholded at 95th percentile
    energy_density = min_incremental_energy / (duration + eps)
    rank_weighted_energy = energy_density / (upward_rank + eps)
    threshold_rank_energy = np.percentile(rank_weighted_energy, 95.0, method='midpoint') + eps
    energy_penalty_mask = (rank_weighted_energy > threshold_rank_energy).astype(float)
    norm_energy_density = robust_minmax_norm(energy_density)
    energy_penalty = norm_energy_density * energy_penalty_mask

    # Fairness: trimmed-mean wait-time scaling for non-urgent tasks only
    # Trim outliers (10% each tail) before normalization to avoid starvation bias
    wait_trim_n = max(1, int(0.1 * N))
    sorted_wait = np.sort(ready_wait_time)
    trimmed_wait = sorted_wait[wait_trim_n:-wait_trim_n] if N > 2 * wait_trim_n else sorted_wait
    wait_min = np.min(trimmed_wait) if len(trimmed_wait) > 0 else 0.0
    wait_max = np.max(trimmed_wait) if len(trimmed_wait) > 0 else 0.0
    wait_rng = wait_max - wait_min + eps
    norm_wait_time = np.clip((ready_wait_time - wait_min) / wait_rng, 0.0, 1.0)
    wait_penalty = (1.0 - is_urgent) * norm_wait_time

    # Uncertainty boost: activated only when both slack is tight AND rank is high
    slack_abs = np.abs(slack) + 1.0
    slack_scale_factor = np.clip(1.0 / slack_abs, 0.1, 10.0)
    rank_gate = robust_minmax_norm(upward_rank)
    uncertainty_boost = uncertainty * slack_scale_factor * rank_gate
    norm_uncertainty_boost = robust_minmax_norm(uncertainty_boost)

    # Remaining work contribution: minor smoothing term
    norm_remaining_work = robust_minmax_norm(remaining_work)

    # Final weighted combination (hierarchical weights from performance analysis)
    score = (
        base_score +
        0.25 * norm_critical_latency +
        0.22 * energy_penalty +
        0.12 * wait_penalty +
        0.09 * norm_uncertainty_boost +
        0.04 * norm_remaining_work
    )

    # Sanitize numerical issues
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.where(is_urgent, -1e12, score)

    # Ensure shape compliance
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
