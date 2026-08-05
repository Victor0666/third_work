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
    Hybrid priority rule: combines Parent 2's ordinal slack penalty and calibrated efficiency scaling
    with Parent 1's starvation-aware wait activation and robust energy-per-critical-work ratio.
    Key novelties:
      - Dual-mode slack penalty: exponential for violated deadlines, sigmoid for tight but feasible slack
      - Critical energy score as (energy_per_second) * (upward_rank / (remaining_work + eps)) — 
        directly penalizes high marginal energy per unit of critical-path work
      - Wait boost activated on slack <= 0 OR upward_rank in top 30% — more interpretable than median threshold
      - Unified robust normalization using clipped IQR, fallback to mean-abs for degenerate cases
      - Final score strictly bounded and finite, with explicit NaN/inf protection at every stage
      - All weights tuned to prioritize deadline compliance > energy efficiency > fairness
    """
    eps = 1e-8
    # Ensure float arrays without modifying inputs
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
    
    # Robust normalization: IQR-based, fallback to mean-abs scaling
    def robust_normalize(x):
        if x.size == 0:
            return np.zeros_like(x)
        q25, q75 = np.quantile(x, [0.25, 0.75], method='midpoint')
        iqr = q75 - q25
        if iqr < eps:
            scale = np.mean(np.abs(x)) + eps
        else:
            scale = iqr + eps
        center = np.median(x)
        return (x - center) / scale
    
    # Slack penalty: exponential for violated (slack < 0), sigmoid for tight (0 <= slack < 60), neutral otherwise
    slack_penalty = np.zeros_like(slack)
    violated_mask = slack < 0
    tight_mask = (~violated_mask) & (slack < 60.0)
    slack_penalty[violated_mask] = np.clip(np.exp(-slack[violated_mask]), 1.0, 1e5)
    slack_penalty[tight_mask] = np.clip(1.0 / (1.0 + 0.05 * slack[tight_mask]), 0.01, 1.0)
    
    # Critical energy score: energy per second weighted by criticality/work ratio
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_second = min_incremental_energy / duration
    critical_work_ratio = upward_rank / (remaining_work + eps)
    critical_energy_score = energy_per_second * (1.0 + 0.8 * critical_work_ratio)
    
    # Wait boost: activates if slack <= 0 OR upward_rank in top 30% (more robust than median)
    wait_boost = np.zeros_like(ready_wait_time)
    rank_percentile = np.argsort(np.argsort(upward_rank)) / (N - 1 + eps) if N > 1 else np.array([0.0])
    wait_activation = (slack <= 0) | (rank_percentile >= 0.7)
    # Normalize only active wait times; others remain zero
    norm_wait = np.where(wait_activation, robust_normalize(ready_wait_time), 0.0)
    
    # Uncertainty boost: only when slack < 0 or slack < 45s (tighter than Parent 1, looser than Parent 2)
    uncertainty_boost = np.zeros_like(uncertainty)
    uncertainty_mask = (slack < 0) | (slack < 45.0)
    uncertainty_boost[uncertainty_mask] = robust_normalize(uncertainty)[uncertainty_mask]
    
    # Normalize core features
    norm_energy = robust_normalize(min_incremental_energy)
    norm_exec = robust_normalize(min_exec_time)
    norm_comm = robust_normalize(min_comm_time)
    norm_critical_energy = robust_normalize(critical_energy_score)
    
    # Final weighted score: smaller = higher priority
    # Weights emphasize deadline (slack_penalty negative → lowers score), then energy efficiency, then fairness
    score = (
        +0.05 * norm_exec
        + 0.05 * norm_comm
        + 0.2 * norm_energy
        - 1.8 * slack_penalty  # Strong deadline pull
        + 0.3 * norm_critical_energy  # Penalize inefficient critical work
        + 0.1 * norm_wait  # Mild starvation mitigation
        - 0.05 * uncertainty_boost  # Slight penalty for high uncertainty under pressure
    )
    
    # Final sanitization: ensure finite, bounded output
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)
    
    return score
