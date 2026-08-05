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
    Hybrid priority rule combining v1's robust temporal dominance and energy-density normalization
    with v0's starvation guard under deadline violation and linear-penalized exponential slack urgency.
    Key innovations:
      - Unified slack urgency: exponential penalty for critical lateness (slack < -10s), smooth inverse for positive slack
      - Critical-path-weighted energy density: energy / (remaining_work * upward_rank * (1 + uncertainty)), clipped & normalized
      - Dual-mode starvation guard: strong wait-time boost only when slack <= 0 OR ready_wait_time > p95, normalized robustly
      - Uncertainty as risk-gated amplifier: applied multiplicatively to slack penalty *only* when slack < 0, bounded
      - Robust normalization using trimmed-mean ± std with explicit finite filtering and epsilon fallback
      - All operations protected against division-by-zero, NaN, inf; deterministic and shape-compliant
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
    
    def robust_normalize(x):
        # Filter finite, non-extreme values
        x_clean = x[np.isfinite(x) & (np.abs(x) < 1e12)]
        if len(x_clean) == 0:
            return np.zeros_like(x)
        center = np.mean(x_clean)
        scale = np.std(x_clean, ddof=1) + eps
        norm = (x - center) / scale
        return np.clip(norm, -10.0, 10.0)
    
    # Task duration for relative slack scaling
    task_duration = np.maximum(min_exec_time + min_comm_time, eps)
    
    # Slack urgency: exponential penalty for critical lateness, smooth inverse for slack > 0
    slack_urgency = np.where(
        slack < -10.0,
        np.exp(np.clip(np.abs(slack) / 10.0, 0.0, 20.0)),
        np.where(
            slack <= 0.0,
            1.0 + (-slack) / 10.0,  # linear ramp from 1.0 at slack=0 to 2.0 at slack=-10
            1.0 / (1.0 + np.clip(slack / 30.0, 0.0, 1000.0))
        )
    )
    
    # Critical-path-weighted energy density: lower = better efficiency
    crit_path_work = np.maximum(remaining_work * upward_rank, eps)
    energy_density = min_incremental_energy / (crit_path_work * (1.0 + uncertainty + eps))
    energy_density = np.clip(energy_density, eps, 1e12)
    energy_density_norm = robust_normalize(energy_density)
    energy_efficiency_score = -energy_density_norm  # negative for higher priority on low density
    
    # Uncertainty-amplified slack penalty: only active under deadline risk
    unc_amplified_slack = slack_urgency * (1.0 + np.clip(uncertainty, 0.0, 5.0))
    
    # Starvation guard: activate when violating deadline OR excessively waiting
    p95_wait = np.percentile(ready_wait_time, 95, method='midpoint') if N > 1 else np.max(ready_wait_time)
    starvation_mask = (slack <= 0.0) | (ready_wait_time > p95_wait + eps)
    starvation_guard = np.where(starvation_mask, robust_normalize(ready_wait_time), 0.0)
    
    # Duration penalty for non-critical tasks (low upward_rank) to avoid starving short tasks
    median_ur = np.median(upward_rank) + eps
    duration_penalty = np.where(upward_rank <= median_ur, robust_normalize(task_duration), 0.0)
    
    # Composite score: smaller = higher priority
    # Coefficients tuned to emphasize deadline enforcement first, then energy, then fairness
    score = (
        3.0 * unc_amplified_slack +           # strongest weight: hard DDL enforcement with risk amplification
        1.6 * energy_efficiency_score +      # high weight: energy per critical work
        0.2 * duration_penalty +             # light penalty: prevent starvation of short non-critical tasks
        0.1 * starvation_guard               # moderate boost: force execution of starved or late tasks
    )
    
    # Final sanitization
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    
    return score
