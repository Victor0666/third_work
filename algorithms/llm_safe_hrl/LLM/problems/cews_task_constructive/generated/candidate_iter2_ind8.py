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
    Hybrid priority rule combining Parent 2's robustness and Parent 1's smooth urgency modeling.
    Key improvements:
    - Uses sigmoid slack urgency (Parent 1) for bounded, differentiable deadline pressure
    - Retains IQR-based robust normalization (Parent 2) for outlier resilience
    - Introduces criticality-weighted energy efficiency: penalizes high energy on high-upward-rank tasks
    - Applies strict slack gating: waiting time & uncertainty boosts only active when slack <= 0
    - Adds starvation guard: boosts long-waiting tasks *only* when they're also critical (upward_rank > median)
    - Normalizes all terms to [-1, 1] before combination to ensure balanced influence
    - All operations protected against NaN/inf/zero via eps and nan_to_num
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

    # Robust IQR-based normalization with fallback to mean-abs scaling
    def robust_normalize(x):
        if x.size == 0:
            return np.zeros_like(x)
        q1, q3 = np.quantile(x, [0.25, 0.75], method='midpoint')
        iqr = q3 - q1
        if iqr < eps:
            scale = np.mean(np.abs(x)) + eps
        else:
            scale = iqr + eps
        return x / scale

    # Smooth, bounded urgency: sigmoid of slack (Parent 1) — avoids infinite gradients
    slack_mean_abs = np.mean(np.abs(slack)) + eps
    slack_urgency = 1.0 / (1.0 + np.exp(-slack / slack_mean_abs))

    # Energy efficiency: energy per second, but weighted by criticality to avoid neglecting vital tasks
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_second = min_incremental_energy / duration
    # Criticality-weighted energy penalty: high energy + high rank = strong penalty
    critical_energy_penalty = energy_per_second * robust_normalize(upward_rank)

    # Leverage remaining work to temper upward_rank inflation on tiny nodes
    work_scale = robust_normalize(remaining_work)
    leveraged_rank = upward_rank * (1.0 + 0.5 * work_scale)

    # Uncertainty boost only when slack <= 0 (strict gating from Parent 2)
    uncertainty_boost = np.where(slack <= 0, robust_normalize(uncertainty), 0.0)

    # Starvation guard: wait boost only if task is both late (slack <= 0) AND critical (rank > median)
    median_rank = np.median(upward_rank) if len(upward_rank) > 1 else 0.0
    wait_activation = (slack <= 0) & (upward_rank > median_rank)
    norm_wait = np.where(wait_activation, robust_normalize(ready_wait_time), 0.0)

    # Normalize all components to [-1, 1] range for balanced linear combination
    def to_unit_range(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x_min, x_max = np.min(x), np.max(x)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return 2.0 * (x - x_min) / (x_max - x_min + eps) - 1.0

    score = (
        +0.35 * to_unit_range(robust_normalize(critical_energy_penalty))   # prioritize energy-efficient critical tasks
        - 1.4 * to_unit_range(slack_urgency)                              # strong deadline pressure (higher weight)
        + 0.2 * to_unit_range(robust_normalize(leveraged_rank))          # reward critical-path progress
        + 0.15 * to_unit_range(uncertainty_boost)                        # risk-aware preemption only when needed
        - 0.1 * to_unit_range(norm_wait)                                 # mild starvation relief for critical late tasks
        + 0.05 * to_unit_range(robust_normalize(min_exec_time))          # slight bias toward fast-executing tasks
        + 0.05 * to_unit_range(robust_normalize(min_comm_time))          # slight bias toward low-comm tasks
    )

    # Final sanitization: ensure finite, deterministic output
    score = np.nan_to_num(score, nan=1e6, posinf=1e6, neginf=-1e6)
    score = np.clip(score, -1e6, 1e6)
    return score
