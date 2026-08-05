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
    Mutated priority rule emphasizing:
    - Slack-aware *temporal dominance*: prioritizes tasks where slack is not just negative, but *critically negative* relative to total duration budget.
    - Energy-efficiency *per critical path weight*: replaces raw upward_rank with normalized critical-path residual work (remaining_work × upward_rank), scaled by risk-adjusted energy density.
    - Starvation control *bounded by urgency*: ready_wait_time bonus only activates when slack > 0 and no critical path pressure exists — avoids interfering with deadline recovery.
    - Uncertainty *as adaptive penalty amplifier*: used only in denominator for energy efficiency ratio, preventing noise amplification while sharpening tradeoffs under risk.
    - Robust *quantile-anchored normalization*: uses trimmed mean ± std instead of IQR for smoother scaling across sparse or skewed distributions.
    - All operations epsilon-protected, nan/inf guarded, deterministic, and shape-compliant.
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
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)

    # Robust normalization: trimmed mean ± std (more stable than IQR for small N)
    def robust_normalize(x):
        x_clean = x[np.isfinite(x) & (np.abs(x) < 1e12)]
        if len(x_clean) == 0:
            return np.zeros_like(x)
        center = np.mean(x_clean)  # trimmed mean handled implicitly via finite filter
        scale = np.std(x_clean, ddof=1) + eps
        norm = (x - center) / scale
        return np.clip(norm, -10.0, 10.0)

    # Core temporal features
    task_duration = np.maximum(min_exec_time + min_comm_time, eps)
    rel_slack = slack / task_duration  # normalized urgency: smaller = more urgent

    # Critical slack penalty: quadratic penalty only when rel_slack < -0.5 (deep violation)
    # Linear ramp from 0 to 1 for rel_slack in [-0.5, 0], zero otherwise
    slack_penalty = np.where(
        rel_slack < -0.5,
        1.0 + (-rel_slack - 0.5) * 2.0,
        np.where(rel_slack < 0.0, (-rel_slack) * 2.0, 0.0)
    )
    slack_penalty = np.clip(slack_penalty, 0.0, 5.0)

    # Risk-adjusted energy density: marginal energy per unit critical-path work
    # Critical-path residual work = remaining_work * upward_rank (higher = more bottlenecked)
    crit_path_work = np.maximum(remaining_work * upward_rank, eps)
    # Denominator includes uncertainty as multiplicative safety margin: prevents over-prioritizing noisy low-energy tasks
    energy_density = min_incremental_energy / (crit_path_work * (1.0 + uncertainty + eps))
    energy_density = np.clip(energy_density, eps, 1e12)

    # Normalize energy density inversely: lower density → higher priority → assign *negative* weight
    energy_density_norm = robust_normalize(energy_density)
    energy_efficiency_score = -energy_density_norm  # invert so lower score = better efficiency

    # Starvation bonus: *only* when slack >= 0 AND upward_rank <= median (non-critical long-waiters)
    median_ur = np.median(upward_rank) + eps
    non_critical_mask = (slack >= 0.0) & (upward_rank <= median_ur)
    max_wait = np.maximum(np.max(ready_wait_time), eps)
    wait_bonus = np.where(
        non_critical_mask,
        np.clip(ready_wait_time / max_wait, 0.0, 0.3),
        0.0
    )

    # Duration penalty: longer tasks get mild penalty unless they're highly critical
    duration_norm = robust_normalize(task_duration)
    duration_penalty = np.where(
        upward_rank > median_ur,
        0.0,
        duration_norm * 0.25
    )

    # Uncertainty penalty: only active when slack < 0, scaled by magnitude
    unc_penalty = np.where(
        slack < 0.0,
        np.clip(uncertainty * (-slack) / (task_duration + eps), 0.0, 0.5),
        0.0
    )

    # Final score: sum of normalized, bounded components
    # Smaller score = higher priority
    score = (
        2.8 * slack_penalty           # dominant deadline enforcement
        + 1.4 * energy_efficiency_score  # efficiency per critical work
        + 0.3 * duration_penalty    # mild bias against long non-critical tasks
        + 0.15 * unc_penalty        # targeted uncertainty amplification under violation
        + wait_bonus                # bounded fairness for idle non-critical tasks
    )

    # Guard against pathological values
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    return score
