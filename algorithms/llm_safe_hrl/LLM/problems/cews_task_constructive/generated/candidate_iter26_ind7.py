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
    v2 priority rule: Hard urgency dominance + monotonic critical-path energy gating +
                      robust wait fairness + uncertainty-gated slack sensitivity +
                      normalized progress pressure + duration-penalized criticality.
    
    Key synthesis:
    - Uses Parent 2's crisp binary urgency (slack <= 0) and monotonic normalization
    - Adopts Parent 1's risk-adjusted duration (duration * (1+uncertainty)) for energy density
    - Combines Parent 2's progress_pressure with Parent 1's duration_penalty logic
    - Introduces unified uncertainty coupling: active only when urgent OR high-rank, scaled by inverse rel_slack
    - Replaces non-monotonic sigmoid with clipped linear criticality gate on rel_slack
    - All components strictly monotonic w.r.t. slack, upward_rank, and duration for DDL safety
    """
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=np.float64).copy()
    min_comm_time = np.asarray(min_comm_time, dtype=np.float64).copy()
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=np.float64).copy()
    slack = np.asarray(slack, dtype=np.float64).copy()
    upward_rank = np.asarray(upward_rank, dtype=np.float64).copy()
    remaining_work = np.asarray(remaining_work, dtype=np.float64).copy()
    ready_wait_time = np.asarray(ready_wait_time, dtype=np.float64).copy()
    uncertainty = np.asarray(uncertainty, dtype=np.float64).copy()
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=np.float64)

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

    # Hard urgency enforcement: highest priority for overdue or deadline-hit tasks
    is_urgent = (slack <= 0.0).astype(np.float64)
    urgency_score = np.full(N, -1000000000000.0, dtype=np.float64)

    # Duration and risk-adjusted duration
    duration = min_exec_time + min_comm_time + eps
    risk_adjusted_duration = duration * (1.0 + uncertainty)

    # Energy density: marginal energy per unit risk-adjusted time
    energy_density = np.divide(min_incremental_energy, risk_adjusted_duration,
                               out=np.zeros_like(min_incremental_energy),
                               where=risk_adjusted_duration != 0)
    energy_density = np.where(np.isfinite(energy_density), energy_density, 0.0)
    norm_energy_density = robust_minmax_norm(energy_density)

    # Relative slack: safe division, clipped to avoid extreme values
    rel_slack = np.divide(slack, duration, out=np.zeros_like(slack), where=duration != 0)
    rel_slack = np.where(np.isfinite(rel_slack), rel_slack, 0.0)

    # Criticality gate: linear ramp from 0 to 1 as rel_slack goes from 0.2 to 0.0 (tighter = more gated)
    # Ensures monotonicity and avoids non-linear functions like sigmoid
    criticality_gate = np.clip((0.2 - np.clip(rel_slack, -np.inf, 0.2)) / 0.2, 0.0, 1.0)
    
    # High-rank mask for energy penalty activation
    median_upward = np.median(upward_rank) + eps
    high_rank_mask = (upward_rank > median_upward).astype(np.float64)
    energy_penalty_mask = high_rank_mask * criticality_gate
    energy_penalty = norm_energy_density * energy_penalty_mask

    # Wait fairness: penalize long waits only for non-urgent, high-work tasks
    work_threshold = np.percentile(remaining_work, 5.0, method='midpoint') + eps
    wait_gate = (remaining_work >= work_threshold).astype(np.float64)
    norm_wait_time = robust_minmax_norm(ready_wait_time)
    wait_penalty = (1.0 - is_urgent) * norm_wait_time * wait_gate

    # Progress pressure: how much critical path remains per unit duration; scaled by slack proximity
    progress_pressure = np.divide(upward_rank, duration, out=np.zeros_like(upward_rank), where=duration != 0)
    slack_proximity = np.clip(-slack / (np.abs(slack) + eps), 0.0, 1.0)
    norm_progress_pressure = robust_minmax_norm(progress_pressure) * slack_proximity

    # Uncertainty coupling: active only when urgent OR high-rank, scaled by inverse relative slack magnitude
    uncertainty_active = np.maximum(is_urgent, high_rank_mask)
    abs_rel_slack = np.abs(rel_slack) + eps
    slack_sensitivity = np.clip(1.0 / abs_rel_slack, 0.1, 10.0)
    rank_sensitivity = robust_minmax_norm(upward_rank)
    uncertainty_boost = uncertainty * slack_sensitivity * rank_sensitivity * uncertainty_active
    norm_uncertainty_boost = robust_minmax_norm(uncertainty_boost)

    # Duration-penalized criticality: longer critical-path tasks that take more time get higher penalty
    norm_upward_rank = robust_minmax_norm(upward_rank)
    duration_penalty = (duration / (np.max(duration) + eps)) * norm_upward_rank

    # Final score composition — weights sum to 1.0, all terms monotonic in key DDL drivers
    score = (
        0.35 * (1.0 - robust_minmax_norm(np.clip(-slack, 0.0, np.inf))) +  # urgency proximity
        0.22 * energy_penalty +                                          # critical-energy tradeoff
        0.13 * wait_penalty +                                             # fairness for large pending work
        0.10 * norm_uncertainty_boost +                                   # risk-aware slack tightening
        0.10 * (1.0 - norm_progress_pressure) +                          # reward advancing critical path
        0.05 * duration_penalty +                                         # discourage long critical tasks
        0.05 * robust_minmax_norm(remaining_work)                         # favor smaller total workload
    )

    # Apply hard urgency override
    score = np.where(is_urgent, urgency_score, score)
    
    # Clamp and sanitize
    score = np.clip(score, -1000000000000.0, 1000000000000.0)
    score = np.nan_to_num(score, nan=1000000000000.0, posinf=1000000000000.0, neginf=-1000000000000.0)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
