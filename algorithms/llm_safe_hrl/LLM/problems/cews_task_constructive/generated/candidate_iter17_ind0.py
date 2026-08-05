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
    v2 priority rule: Lexicographic urgency-energy dominance with robust fairness & uncertainty-gated criticality.
    
    Key innovations:
    - Hybrid urgency: hard bias for negative slack + trimmed-mean-centered sigmoid for near-deadline tasks
    - Critical-path energy penalty *only* when both slack is tight AND uncertainty is high (uncertainty-aware gating)
    - Starvation boost uses dual-gating: wait-per-work z-score > 1.5 AND remaining_work in top 30% quantile
    - Uncertainty-normalized slack: rel_slack scaled by (1 + uncertainty / task_duration) to suppress risky late tasks
    - Robust trimmed-minmax normalization (10% trim) for all metrics — stable under small-N and outliers
    - All divisions guarded; NaN/inf replaced deterministically; no unbounded ops; shape-invariant
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

    def robust_trimmed_minmax(x):
        if x.size == 0:
            return np.zeros_like(x)
        x = x.astype(float)
        x_sorted = np.sort(x)
        trim_n = max(1, int(0.1 * len(x_sorted)))
        x_trimmed = x_sorted[trim_n:-trim_n] if len(x_sorted) > 2 * trim_n else x_sorted
        x_min, x_max = np.min(x_trimmed), np.max(x_trimmed)
        rng = x_max - x_min + eps
        return np.clip((x - x_min) / rng, 0.0, 1.0)

    # Task duration and robust relative slack
    task_duration = min_exec_time + min_comm_time + eps
    rel_slack = np.divide(slack, task_duration, out=np.full_like(slack, 0.0), where=task_duration != 0)
    rel_slack = np.where(np.isfinite(rel_slack), rel_slack, 0.0)

    # Uncertainty-augmented slack: penalize high-uncertainty tasks near deadline
    dur_uncertainty = np.divide(uncertainty, task_duration + eps, out=np.zeros_like(uncertainty), where=task_duration + eps != 0)
    aug_rel_slack = rel_slack / (1.0 + dur_uncertainty + eps)

    # Trimmed-mean center for stable sigmoid urgency scaling
    finite_aug_slack = aug_rel_slack[np.isfinite(aug_rel_slack)]
    slack_center = np.mean(finite_aug_slack) if len(finite_aug_slack) > 0 else 0.0

    # Hard urgency bias + smooth sigmoid urgency
    has_negative_slack = (slack <= 0.0).astype(float)
    urgency_bias = np.where(has_negative_slack, -1.0, 0.0)
    urgency_sigmoid = 1.0 / (1.0 + np.exp(6.0 * (aug_rel_slack - slack_center)))
    urgency = np.where(has_negative_slack, 1.0, urgency_sigmoid)

    # Lateness penalty (for violation magnitude)
    lateness_penalty = np.where(slack < 0.0, np.clip(-slack / (np.median(task_duration) + eps), 0.0, 4.0), 0.0)

    # Critical-path latency pressure: only activated under tight slack AND high uncertainty
    tight_slack_mask = (aug_rel_slack <= 0.2).astype(float)
    high_uncert_mask = (dur_uncertainty >= np.quantile(dur_uncertainty, 0.7, method='midpoint')).astype(float)
    cp_latency_penalty = upward_rank * np.maximum(0.0, -slack) / (task_duration + eps) * tight_slack_mask * high_uncert_mask

    # Energy per work, critically weighted only under pressure
    energy_per_work = np.divide(min_incremental_energy, remaining_work + eps, out=np.full_like(min_incremental_energy, eps), where=remaining_work + eps != 0)
    crit_weight_factor = 1.0 + np.clip(upward_rank / (np.median(upward_rank + eps) + eps), 0.0, 2.0)
    base_crit_energy = energy_per_work * crit_weight_factor
    # Scale energy cost only when slack is tight AND uncertainty is elevated
    crit_weighted_energy = base_crit_energy * tight_slack_mask * high_uncert_mask

    # Starvation boost: dual-gated fairness
    wait_per_work = np.divide(ready_wait_time, remaining_work + eps, out=np.full_like(ready_wait_time, 0.0), where=remaining_work + eps != 0)
    work_threshold = np.quantile(remaining_work, 0.7, method='midpoint') + eps
    wait_median = np.median(wait_per_work)
    wait_std = np.std(wait_per_work) + eps
    wait_z_score = np.divide(wait_per_work - wait_median, wait_std, out=np.zeros_like(wait_per_work), where=wait_std != 0)
    starvation_gate = (
        (remaining_work >= work_threshold).astype(float) *
        (wait_z_score > 1.5).astype(float)
    )
    norm_wait_time = robust_trimmed_minmax(ready_wait_time)
    wait_boost = norm_wait_time * starvation_gate * 0.3

    # Normalize all components robustly
    norm_urgency = robust_trimmed_minmax(urgency)
    norm_lateness = robust_trimmed_minmax(lateness_penalty)
    norm_cp_latency = robust_trimmed_minmax(cp_latency_penalty)
    norm_energy = robust_trimmed_minmax(crit_weighted_energy)
    norm_upward = robust_trimmed_minmax(upward_rank)
    norm_uncertainty = robust_trimmed_minmax(dur_uncertainty)

    # Final score: smaller = higher priority
    # Lexicographic emphasis via coefficient ordering: urgency dominates, then latency, energy, uncertainty
    score = (
        -3.0 * norm_urgency +           # Strongest pull toward urgent tasks
         1.2 * norm_lateness +          # Moderate penalty for violation magnitude
         1.0 * norm_energy +            # Energy minimization under pressure
         0.4 * norm_uncertainty +       # Slight penalty for high-risk near-deadline tasks
        -0.5 * norm_upward +            # Mild preference for lower critical-path rank (less vital first)
         0.3 * wait_boost +             # Fairness boost for starved heavy tasks
         0.9 * norm_cp_latency +        # Priority for critical tasks under combined pressure
         urgency_bias                   # Hard bias pushes negative-slack tasks to top
    )

    # Final sanitization
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
