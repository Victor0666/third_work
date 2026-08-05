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
    Self-evolved priority rule v2: Strict deadline-first enforcement with adaptive risk-aware energy efficiency,
    starvation-aware fairness, and robust bounded normalization.
    
    Key integrations:
      - Hard urgency override for *any* negative slack (v2 strength) + urgency sigmoid for borderline cases (v1)
      - Decoupled lateness penalty (v2) + uncertainty-modulated penalty only under tight slack (v2)
      - Criticality-weighted energy density per unit work (v2), normalized via robust_minmax (v2), not raw scaling
      - Starvation gating using wait-time percentile *and* slack safety guard (v0 insight refined in v2: rel_slack > -0.02)
      - Robust_minmax with fallbacks (v2) replaces unstable quantile/IQR scaling → deterministic ordering & low variance
      - Added latency-aware energy efficiency term: energy_per_duration = energy / (duration + eps), scaled by uncertainty only when safe
      - All operations guarded against zero, NaN, inf; clipped to finite bounds; deterministic for identical inputs.
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

    def robust_minmax(x):
        if len(x) == 0:
            return x
        x_min, x_max = np.min(x), np.max(x)
        rng = x_max - x_min + eps
        return np.clip((x - x_min) / rng, 0.0, 1.0)

    # Core derived metrics
    task_duration = min_exec_time + min_comm_time + eps
    rel_slack = slack / task_duration
    has_negative_slack = (slack < 0.0).astype(float)
    
    # Urgency: hard override for any violation + smooth sigmoid for borderline urgency
    slack_threshold = np.quantile(rel_slack, 0.3) if N > 1 else np.median(rel_slack)
    urgency_sigmoid = 1.0 / (1.0 + np.exp(6.0 * (rel_slack - slack_threshold)))
    urgency = np.where(has_negative_slack, 1.0, urgency_sigmoid)
    
    # Lateness penalty: only active when slack < 0, bounded and normalized
    lateness_penalty = np.where(slack < 0.0, np.clip(-slack / (np.median(task_duration) + eps), 0.0, 4.0), 0.0)
    
    # Tight-slack uncertainty boost: only when relative slack <= 0.1 (avoids over-penalizing robust tasks)
    tight_slack_mask = (rel_slack <= 0.1).astype(float)
    dur_uncertainty = np.where(task_duration > eps, uncertainty / task_duration, 0.0)
    uncertainty_boost = dur_uncertainty * tight_slack_mask
    
    # Energy efficiency: inverse energy density per unit work, weighted by criticality
    energy_per_work = np.maximum(min_incremental_energy / (remaining_work + eps), eps)
    rank_scale = 1.0 + upward_rank / (np.median(upward_rank + eps) + eps)
    crit_weighted_energy = energy_per_work * rank_scale
    
    # Latency-aware energy density: energy per duration, modulated by uncertainty only when slack is safe
    energy_per_duration = min_incremental_energy / task_duration
    safe_slack_mask = (rel_slack > -0.02).astype(float)
    energy_latency_risk = energy_per_duration * (1.0 + 0.7 * np.clip(uncertainty, 0.0, 1.0)) * safe_slack_mask
    
    # Fairness: starvation-aware wait boost — only when not urgent AND sufficient work remains
    wait_gate = (rel_slack > -0.02).astype(float) * (remaining_work > np.quantile(remaining_work, 0.2) + eps).astype(float)
    norm_wait_time = robust_minmax(ready_wait_time)
    wait_boost = norm_wait_time * wait_gate * 0.25
    
    # Normalize all components deterministically
    norm_urgency = robust_minmax(urgency)
    norm_lateness = robust_minmax(lateness_penalty)
    norm_energy = robust_minmax(crit_weighted_energy)
    norm_uncertainty = robust_minmax(uncertainty_boost)
    norm_energy_latency = robust_minmax(energy_latency_risk)
    norm_upward = robust_minmax(upward_rank)
    
    # Final score: smaller = higher priority
    # Weights sum to ~1.0 for interpretability; urgency dominates; lateness and energy penalize
    score = (
        -3.0 * norm_urgency           # Highest priority: meet DDL
        + 1.3 * norm_lateness         # Penalty for violations
        + 0.9 * norm_energy           # Criticality-weighted energy inefficiency
        + 0.4 * norm_uncertainty      # Risk amplification under tight slack
        + 0.3 * norm_energy_latency   # Latency-aware energy cost under safe slack
        - 0.4 * norm_upward           # Prefer higher criticality (lower score)
        + 0.15 * wait_boost           # Boost long-waiting non-critical tasks
    )
    
    # Final safeguards
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
