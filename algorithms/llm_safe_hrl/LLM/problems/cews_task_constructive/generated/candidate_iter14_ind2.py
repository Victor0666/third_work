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
    v2 priority rule: Hybrid of Parent 1's zero-tolerance urgency & Parent 2's robust slack-aware adaptivity.
    Combines crisp hard-DDL dominance with smooth risk-aware scaling; replaces brittle thresholds with trimmed statistics;
    introduces criticality-normalized energy penalty gated by slack-driven urgency; enhances fairness via wait-per-work z-score;
    uses strict hierarchy enforcement (urgency > latency > energy > uncertainty > fairness) with deterministic clipping.
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

    def robust_minmax(x):
        if x.size == 0:
            return np.zeros_like(x)
        x_sorted = np.sort(x)
        trim_n = max(1, int(0.1 * len(x_sorted)))
        x_trimmed = x_sorted[trim_n:-trim_n] if len(x_sorted) > 2 * trim_n else x_sorted
        x_min, x_max = np.min(x_trimmed), np.max(x_trimmed)
        rng = x_max - x_min + eps
        return np.clip((x - x_min) / rng, 0.0, 1.0)

    # Core temporal metrics
    task_duration = min_exec_time + min_comm_time + eps
    rel_slack = np.divide(slack, task_duration, out=np.full_like(slack, np.inf, dtype=float), where=task_duration!=0)
    rel_slack = np.where(np.isfinite(rel_slack), rel_slack, 0.0)

    # Crisp zero-tolerance urgency: immediate highest priority if slack <= 0
    has_negative_slack = (slack <= 0.0).astype(float)
    urgency_bias = np.where(has_negative_slack, -1.0, 0.0)

    # Lateness penalty: linear penalty for negative slack, capped at 4.0
    lateness_penalty = np.where(slack < 0.0, np.clip(-slack / (np.median(task_duration) + eps), 0.0, 4.0), 0.0)

    # Critical-path latency penalty: prioritize critical tasks under deadline pressure
    cp_latency_penalty = upward_rank * np.maximum(0.0, -slack) / (task_duration + eps)

    # Energy efficiency: risk-adjusted energy per critical work, scaled by slack pressure
    energy_per_work = np.divide(min_incremental_energy, remaining_work + eps, out=np.full_like(min_incremental_energy, eps), where=(remaining_work + eps)!=0)
    base_crit_energy = energy_per_work * (1.0 + upward_rank / (np.median(upward_rank + eps) + eps))
    slack_scale_factor = 1.0 + np.maximum(0.0, -slack) / (task_duration + eps)
    crit_weighted_energy = base_crit_energy * slack_scale_factor

    # Uncertainty boost: only activated when near deadline (rel_slack <= 0.25)
    tight_slack_mask = (rel_slack <= 0.25).astype(float)
    dur_uncertainty = np.divide(uncertainty, task_duration + eps, out=np.zeros_like(uncertainty), where=(task_duration + eps)!=0)
    uncertainty_boost = dur_uncertainty * tight_slack_mask

    # Fairness guard: activate wait boost only for non-urgent tasks with high relative wait-per-work
    wait_per_work = np.divide(ready_wait_time, remaining_work + eps, out=np.full_like(ready_wait_time, 0.0), where=(remaining_work + eps)!=0)
    wait_median = np.median(wait_per_work)
    wait_std = np.std(wait_per_work) + eps
    wait_z_threshold = wait_median + 1.5 * wait_std
    wait_gate = (slack > 0.0).astype(float) * (wait_per_work > wait_z_threshold).astype(float)
    norm_wait_time = robust_minmax(ready_wait_time)
    wait_boost = norm_wait_time * wait_gate * 0.25

    # Normalize all components robustly
    norm_lateness = robust_minmax(lateness_penalty)
    norm_cp_latency = robust_minmax(cp_latency_penalty)
    norm_energy = robust_minmax(crit_weighted_energy)
    norm_uncertainty = robust_minmax(uncertainty_boost)
    norm_upward = robust_minmax(upward_rank)

    # Final score: strict hierarchy with urgency bias dominating
    # Base score starts at 1.0, then subtracts urgency bias (-1.0 → 0.0 base), then adds weighted penalties
    score = (
        1.0 + urgency_bias +
        1.3 * norm_lateness +
        1.0 * norm_energy +
        0.3 * norm_uncertainty -
        0.4 * norm_upward +
        0.25 * wait_boost +
        0.8 * norm_cp_latency
    )

    # Ensure finite output and clamp extremes
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)

    return score
