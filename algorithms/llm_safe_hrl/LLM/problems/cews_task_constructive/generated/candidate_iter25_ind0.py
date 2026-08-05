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
    v2 priority rule: Hard urgency dominance + risk-normalized energy density + adaptive starvation relief +
                      uncertainty-gated slack sensitivity + criticality-weighted duration penalty.
    
    Key mutations vs v1:
    - Replaces heuristic energy_penalty_mask with *continuous* criticality-gating via sigmoid on (upward_rank * rel_slack)
    - Introduces risk-normalized energy density: energy / (duration * (1 + uncertainty)) — preserves fidelity under volatility
    - Uses *inverse exponential slack scaling*: exp(-slack / (duration + eps)) — sharp near-deadline pressure, smooth far-out
    - Starvation relief now gated by *both* low remaining_work percentile AND high uncertainty (to prioritize uncertain small tasks)
    - Removes percentile-based work_threshold; uses robust 5%-tile normalized wait_ratio instead
    - Adds duration penalty term weighted by upward_rank to penalize long critical-path tasks delaying descendants
    - All normalization uses clipped percentiles + epsilon-guarded min-max; no z-score or fragile statistics
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
        p01 = np.percentile(x, 1.0)
        p99 = np.percentile(x, 99.0)
        x_clipped = np.clip(x, p01, p99)
        x_min = np.min(x_clipped)
        x_max = np.max(x_clipped)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x_clipped - x_min) / (x_max - x_min + eps)

    # Absolute urgency enforcement: overdue tasks get guaranteed min score
    is_urgent = (slack <= 0.0).astype(float)
    
    # Base duration: exec + comm + epsilon for stability
    duration = min_exec_time + min_comm_time + eps
    
    # Risk-normalized energy density: joules per second per unit risk
    # Preserves marginal energy signal while dampening under high uncertainty
    risk_adjusted_duration = duration * (1.0 + uncertainty)
    energy_density = np.divide(min_incremental_energy, risk_adjusted_duration,
                               out=np.zeros_like(min_incremental_energy),
                               where=risk_adjusted_duration != 0)
    energy_density = np.where(np.isfinite(energy_density), energy_density, 0.0)
    
    # Inverse exponential slack sensitivity: sharp near deadline, flat far out
    # Avoids division-by-zero and NaN in log/exp domains
    slack_pressure = np.exp(-np.clip(slack / (duration + eps), -10.0, 10.0))
    
    # Continuous criticality gating: sigmoid of (upward_rank * rel_slack) → high when both large
    rel_slack = np.divide(slack, duration, out=np.zeros_like(slack), where=duration != 0)
    rel_slack = np.where(np.isfinite(rel_slack), rel_slack, 0.0)
    criticality_gate_input = upward_rank * np.clip(rel_slack, 0.0, 1.0)  # avoid negative slack amplification
    criticality_gate = 1.0 / (1.0 + np.exp(-(criticality_gate_input - 0.5)))  # centered sigmoid
    
    # Normalized terms
    norm_energy_density = robust_minmax_norm(energy_density)
    norm_upward_rank = robust_minmax_norm(upward_rank)
    norm_remaining_work = robust_minmax_norm(remaining_work)
    norm_ready_wait = robust_minmax_norm(ready_wait_time)
    
    # Starvation relief: activate only for tasks with low remaining_work (top 10% smallest) AND high uncertainty
    work_percentile = np.searchsorted(np.sort(remaining_work), remaining_work) / max(N, 1)
    low_work_mask = (work_percentile <= 0.1).astype(float)
    high_uncertainty_mask = (uncertainty >= np.percentile(uncertainty, 90.0)).astype(float)
    starvation_relief = norm_ready_wait * low_work_mask * high_uncertainty_mask
    
    # Duration penalty: penalize long-duration tasks on critical paths
    duration_penalty = (duration / (np.max(duration) + eps)) * norm_upward_rank
    
    # Base score composition (smaller = better)
    base_score = (
        0.32 * norm_energy_density * criticality_gate +          # gated energy minimization
        0.25 * slack_pressure +                                 # urgency pressure (↑ near deadline)
        0.18 * duration_penalty +                               # critical-path delay penalty
        0.10 * (1.0 - starvation_relief) +                      # anti-starvation: subtract relief from penalty
        0.08 * norm_remaining_work +                            # favor small remaining work (early exit bias)
        0.07 * uncertainty                                      # mild uncertainty boost (for exploration)
    )
    
    # Apply hard urgency override
    score = np.where(is_urgent, -1e12, base_score)
    
    # Final clipping and NaN/inf cleanup
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
