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
    v2 priority rule: Hybrid deadline-feasibility + risk-aware energy optimization with:
      - Adaptive lateness penalty scaled by |slack| for fine-grained urgency differentiation
      - Slack-scaled critical timing to suppress premature scheduling under tight positive slack
      - Energy-efficiency boost *only* when slack is sufficiently large (rel_slack > 0.5)
      - Starvation-aware fairness gated by *both* slack > 0 AND remaining_work > median
      - Uncertainty coupling conditioned on triple feasibility: slack > 0 ∧ upward_rank > median ∧ uncertainty > median
      - Robust normalization using clipped percentile-based minmax with N=1 safety and finite-domain fallbacks
      - Explicit zero-slack gating (not just negative) to enforce hard deadline boundary
      - Inverted energy term under safe slack to actively promote low-energy tasks without violating DDL
      - Relative age normalization using critical path length estimate for starvation pressure
      - All operations guarded against div-by-zero, NaN, inf; deterministic and side-effect free
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
        if N == 1:
            return np.zeros_like(x)
        p01 = np.percentile(x, 1.0, method='lower')
        p99 = np.percentile(x, 99.0, method='higher')
        x_clipped = np.clip(x, p01, p99)
        x_min = np.min(x_clipped)
        x_max = np.max(x_clipped)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x_clipped - x_min) / (x_max - x_min + eps)

    # Lateness handling: zero-slack is urgent (not just negative), scale penalty by |slack|
    is_urgent = slack <= eps
    abs_slack = np.abs(slack)
    lateness_penalty = np.where(is_urgent, -1e12 * (1.0 + 0.1 * abs_slack), 0.0)

    # Duration and critical timing
    duration = min_exec_time + min_comm_time + eps
    critical_timing = duration * upward_rank
    norm_critical_timing = robust_minmax_norm(critical_timing)

    # Relative slack for scaling and gating
    rel_slack = np.divide(slack, duration, out=np.zeros_like(slack), where=duration != 0)
    rel_slack = np.nan_to_num(rel_slack, nan=0.0, posinf=0.0, neginf=0.0)

    # Slack-scaled critical timing: suppress bias when slack is tight but positive
    slack_scale = np.clip(1.0 - rel_slack, 0.0, 1.0)
    scaled_critical_timing = norm_critical_timing * slack_scale

    # Energy density (J/s) — use duration, not effective_duration, to avoid circularity with uncertainty
    energy_density = np.divide(min_incremental_energy, duration, out=np.zeros_like(min_incremental_energy), where=duration != 0)
    energy_density = np.nan_to_num(energy_density, nan=0.0, posinf=0.0, neginf=0.0)
    norm_energy_density = robust_minmax_norm(energy_density)

    # Energy-efficiency boost only when slack is ample: promotes low-energy tasks *without* compromising deadline
    efficiency_boost_mask = (slack > eps) & (rel_slack > 0.5)
    efficiency_boost = -0.25 * norm_energy_density * efficiency_boost_mask

    # Fairness: starvation-aware wait pressure normalized by critical path length estimate
    cp_length_estimate = duration * upward_rank + eps
    relative_age = np.divide(ready_wait_time, cp_length_estimate, out=np.zeros_like(ready_wait_time), where=cp_length_estimate != 0)
    relative_age = np.nan_to_num(relative_age, nan=0.0, posinf=0.0, neginf=0.0)
    relative_age = np.clip(relative_age, 0.0, 10.0)
    norm_relative_age = robust_minmax_norm(relative_age)
    rw_median = np.median(remaining_work) if N > 1 else remaining_work[0]
    fairness_mask = (slack > eps) & (remaining_work > rw_median)
    fairness_boost = norm_relative_age * fairness_mask

    # Uncertainty coupling only under triple feasibility: avoids amplifying risk on non-critical/low-risk tasks
    ur_median = np.median(upward_rank) if N > 1 else upward_rank[0]
    unc_median = np.median(uncertainty) if N > 1 else uncertainty[0]
    unc_mask = (slack > eps) & (upward_rank > ur_median) & (uncertainty > unc_median)
    norm_uncertainty = robust_minmax_norm(uncertainty)
    unc_coupling = norm_uncertainty * unc_mask

    # Remaining work importance (normalized)
    norm_remaining_work = robust_minmax_norm(remaining_work)

    # Final score composition: smaller = higher priority
    # Critical path weight dominates under urgency; energy/fairness/uncertainty modulate only under safety
    score = (
        0.48 * scaled_critical_timing +
        0.22 * norm_remaining_work +
        0.12 * robust_minmax_norm(uncertainty) * unc_mask +  # explicit unc term before coupling
        0.10 * fairness_boost +
        0.05 * unc_coupling +
        efficiency_boost
    )

    # Apply lateness penalty (overrides all other considerations for urgent/late tasks)
    score = lateness_penalty + score

    # Final cleanup: ensure finite, bounded, shape-correct output
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
