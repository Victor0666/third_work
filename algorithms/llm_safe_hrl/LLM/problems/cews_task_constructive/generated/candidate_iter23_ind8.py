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
    v2 priority rule: Hard urgency dominance + critical-path energy gating + adaptive fairness +
                      uncertainty-aware slack normalization + calibrated starvation rescue +
                      robust dynamic-range preservation.
    
    Key synthesis improvements:
    - Guarantees top priority for urgent tasks (slack <= 0) via -1e12 score (Parent 2 strength)
    - Applies energy penalty *only* on critical path (upward_rank > 75th percentile) AND tight slack (rel_slack <= 0.3) (Parent 2)
    - Adds calibrated starvation rescue: wait-time boost gated by upward_rank percentile *and* positive slack to avoid biasing late tasks
    - Replaces minmax with robust z-score clipping (±6σ) for fine-grained discrimination among non-urgent tasks (Parent 1 insight)
    - Introduces slack-proximity scaling for energy penalty: tighter slack → stronger energy weighting within safe region
    - Uses uncertainty-weighted urgency amplification only when slack > 0, preventing over-penalization of already-late tasks
    - Final weights prioritize DDL feasibility (0.42), critical-path synergy (0.28), energy-risk (0.15), fairness (0.10), uncertainty (0.05)
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
    
    # Robust z-score clipping: center at median, scale by MAD, clip ±6σ
    def robust_zclip(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, -1e12, 1e12)
        if N == 1:
            return np.array([0.0])
        center = np.median(x)
        mad = np.median(np.abs(x - center)) + eps
        z = (x - center) / mad
        return np.clip(z, -6.0, 6.0)
    
    duration = min_exec_time + min_comm_time + eps
    rel_slack = np.divide(slack, duration, out=np.zeros_like(slack), where=duration != 0)
    rel_slack = np.where(np.isfinite(rel_slack), rel_slack, 0.0)
    
    # Urgent tasks get absolute highest priority
    is_urgent = (slack <= 0.0)
    score = np.full(N, 1.0, dtype=float)
    score = np.where(is_urgent, -1000000000000.0, score)
    
    # Only compute non-urgent components for non-urgent tasks
    non_urgent_mask = ~is_urgent
    if not np.any(non_urgent_mask):
        score = np.nan_to_num(score, nan=1000000000000.0, posinf=1000000000000.0, neginf=-1000000000000.0)
        assert score.shape == (N,)
        return score
    
    # Critical-path latency importance (normalized upward rank × duration)
    norm_ur = robust_zclip(upward_rank)
    critical_latency = duration * (1.0 + 0.6 * np.clip(norm_ur, 0.0, 6.0))
    
    # Energy density: incremental energy per unit time
    energy_density = np.divide(min_incremental_energy, duration, out=np.zeros_like(min_incremental_energy), where=duration != 0)
    energy_density = np.nan_to_num(energy_density, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Critical-path + tight-slack energy penalty mask
    rank_threshold = np.percentile(upward_rank, 75.0) + eps
    tight_slack_mask = (rel_slack <= 0.3)
    high_rank_mask = (upward_rank > rank_threshold)
    energy_penalty_mask = tight_slack_mask & high_rank_mask
    
    # Slack-proximity scaling for energy penalty: tighter slack → higher weight in safe region
    slack_scale = np.where(non_urgent_mask, np.clip(1.0 - rel_slack, 0.0, 1.0), 0.0)
    energy_penalty = robust_zclip(energy_density) * energy_penalty_mask * slack_scale
    
    # Starvation rescue: wait-time boost gated by criticality (upward_rank > 70th percentile) AND slack > 0 AND work > 5th percentile
    work_threshold = np.percentile(remaining_work, 5.0) + eps
    ur_percentile = np.percentile(upward_rank, 70.0) if N > 1 else upward_rank[0]
    is_starvable = (
        (ready_wait_time > 0.5 * duration) &
        (upward_rank >= ur_percentile) &
        (slack > 0.0) &
        (remaining_work >= work_threshold)
    )
    norm_wait_ratio = np.divide(ready_wait_time, duration + eps, out=np.zeros_like(ready_wait_time), where=duration + eps != 0)
    starvation_boost = np.where(is_starvable, np.clip(norm_wait_ratio, 0.0, 3.0), 0.0)
    
    # Uncertainty amplification: only active when slack > 0 and upward_rank > median
    ur_median = np.median(upward_rank) if N > 1 else upward_rank[0]
    unc_amplifier = np.where(
        non_urgent_mask & (slack > 0.0) & (upward_rank > ur_median),
        1.0 + 0.4 * uncertainty,
        1.0
    )
    
    # Normalize components
    norm_critical_latency = robust_zclip(critical_latency)
    norm_starvation = robust_zclip(starvation_boost)
    norm_uncertainty = robust_zclip(uncertainty)
    
    # Assemble non-urgent score: lower is better
    non_urgent_score = (
        0.42 * norm_critical_latency +
        0.28 * (-robust_zclip(remaining_work)) +  # Favor larger remaining work (critical path)
        0.15 * energy_penalty +
        0.10 * norm_starvation +
        0.05 * (norm_uncertainty * unc_amplifier)
    )
    
    # Apply to non-urgent subset
    score = np.where(non_urgent_mask, score + non_urgent_score, score)
    
    # Final cleanup
    score = np.nan_to_num(score, nan=1000000000000.0, posinf=1000000000000.0, neginf=-1000000000000.0)
    score = np.clip(score, -1000000000000.0, 1000000000000.0)
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
