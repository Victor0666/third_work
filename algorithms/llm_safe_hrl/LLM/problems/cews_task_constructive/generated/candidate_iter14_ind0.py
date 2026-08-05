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
    Self-evolved priority rule v2: Deadline-robust critical-path leverage + adaptive risk-energy coupling + z-score fairness.
    
    Key synthesis & innovations:
    - Combines Parent 2's trimmed-mean robust centrality and slack-aware energy scaling with Parent 1's monotonic deadline penalty (no sigmoid fragility)
    - Replaces sigmoid urgency with linear urgency + bounded latency penalty for deterministic ordering under perturbation
    - Uses critical-path leverage (upward_rank / (remaining_work + eps)) instead of raw upward_rank → better work-normalized impact signal
    - Introduces *risk-gated energy efficiency*: energy_per_work scaled by (1 + max(0, -slack)/task_duration) only when slack < 0, avoiding over-penalization
    - Fairness via robust z-score (median + 1.5*MAD) on wait_ratio, activated only when slack >= 0 to avoid interfering with urgent tasks
    - All normalizations use trimmed-minmax with alpha=0.1; all divisions guarded; NaN/inf handled deterministically
    - Final convex combination preserves interpretability and monotonicity guarantees
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
    
    def robust_trimmed_minmax(x, alpha=0.1):
        if x.size == 0:
            return np.zeros_like(x)
        x_sorted = np.sort(x)
        trim_n = max(1, int(alpha * len(x_sorted)))
        x_trimmed = x_sorted[trim_n:-trim_n] if len(x_sorted) > 2 * trim_n else x_sorted
        x_min, x_max = np.min(x_trimmed), np.max(x_trimmed)
        rng = x_max - x_min + eps
        return np.clip((x - x_min) / rng, 0.0, 1.0)
    
    # Core metrics
    task_duration = min_exec_time + min_comm_time + eps
    neg_slack = np.maximum(-slack, 0.0)
    rel_slack = slack / task_duration
    
    # Monotonic deadline penalty: linear urgency + bounded lateness penalty
    deadline_penalty = neg_slack * (1.0 + 0.8 * np.clip(uncertainty, 0.0, 3.0))
    lateness_penalty = np.where(slack < 0.0, np.clip(neg_slack / (np.median(task_duration) + eps), 0.0, 4.0), 0.0)
    
    # Critical-path leverage: impact per unit work
    cp_leverage = upward_rank / (remaining_work + eps)
    norm_cp_leverage = robust_trimmed_minmax(cp_leverage)
    
    # Risk-gated energy efficiency: only penalize high energy under pressure
    energy_per_work = np.maximum(min_incremental_energy / (remaining_work + eps), eps)
    slack_scale_factor = 1.0 + np.where(slack < 0.0, neg_slack / task_duration, 0.0)
    risk_weighted_energy = energy_per_work * slack_scale_factor
    norm_energy = robust_trimmed_minmax(risk_weighted_energy)
    
    # Fairness: starvation boost only for non-urgent tasks (slack >= 0) using robust z-score (MAD-based)
    wait_ratio = ready_wait_time / (remaining_work + eps)
    wait_median = np.median(wait_ratio)
    abs_dev = np.abs(wait_ratio - wait_median)
    mad = np.median(abs_dev) + eps
    wait_z_score = (wait_ratio - wait_median) / mad
    starvation_gate = (slack >= 0.0).astype(float) * (wait_z_score > 1.5).astype(float)
    norm_wait_time = robust_trimmed_minmax(ready_wait_time)
    starvation_boost = norm_wait_time * starvation_gate * 0.25
    
    # Uncertainty coupling: only in tight-slack regime (rel_slack <= 0.25)
    tight_slack_mask = (rel_slack <= 0.25).astype(float)
    dur_uncertainty = np.where(task_duration > eps, uncertainty / task_duration, 0.0)
    uncertainty_boost = dur_uncertainty * tight_slack_mask
    norm_uncertainty = robust_trimmed_minmax(uncertainty_boost)
    
    # Normalize components
    norm_deadline = robust_trimmed_minmax(deadline_penalty)
    norm_lateness = robust_trimmed_minmax(lateness_penalty)
    norm_cp_latency = robust_trimmed_minmax(upward_rank * np.maximum(0.0, -slack) / (task_duration + eps))
    
    # Final convex combination: prioritize deadline compliance first, then energy, fairness, structure
    score = (
        0.42 * norm_deadline +
        0.20 * norm_lateness +
        0.16 * norm_energy +
        0.09 * norm_cp_latency +
        0.07 * norm_uncertainty +
        0.04 * (1.0 - norm_cp_leverage) +  # higher leverage → lower score
        0.02 * starvation_boost
    )
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
