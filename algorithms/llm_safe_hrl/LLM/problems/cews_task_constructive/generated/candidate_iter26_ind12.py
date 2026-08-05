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
    # Robust input cleaning: convert to float, handle NaN/inf/neg-inf
    eps = 1e-08
    tau = 1.0
    def clean(x):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    
    min_exec_time = clean(min_exec_time)
    min_comm_time = clean(min_comm_time)
    min_incremental_energy = clean(min_incremental_energy)
    slack = clean(slack)
    upward_rank = clean(upward_rank)
    remaining_work = clean(remaining_work)
    ready_wait_time = clean(ready_wait_time)
    uncertainty = clean(uncertainty)
    
    # Risk-adjusted urgency: tanh-based, penalizing high uncertainty near deadline
    risk_adjusted_slack = slack - 2.0 * uncertainty
    urgency_raw = np.tanh(risk_adjusted_slack / (tau + eps))
    # Map [-1,1] → [0,1]: higher urgency → lower score (priority)
    urgency_score = (1.0 - urgency_raw) / 2.0
    
    # MAD-based normalization with zero-variance fallback and clipping
    def normalize_mad(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        median_val = np.median(x)
        mad = np.median(np.abs(x - median_val))
        if mad < eps:
            return np.zeros_like(x)
        normed = (x - median_val) / (mad + eps)
        return np.clip(normed, -4.0, 4.0)
    
    # Criticality pressure: only activated when slack < 0 (deadline at risk)
    critical_mask = (slack < 0.0).astype(float)
    # Normalize remaining_work to avoid scale dominance; use upward_rank as importance weight
    norm_remaining_work = remaining_work / (np.max(remaining_work + eps) + eps)
    critical_base = upward_rank * norm_remaining_work
    norm_critical = normalize_mad(critical_base)
    critical_term = 2.5 * norm_critical * critical_mask
    
    # Adaptive energy efficiency: prioritize low energy-per-unit-work only when slack > 0
    total_latency = min_exec_time + min_comm_time + eps
    seer_ratio = min_incremental_energy / total_latency
    energy_base = 1.0 / (seer_ratio + eps)  # Higher = more energy-efficient
    energy_gate = (slack > 0.0).astype(float)
    norm_energy = normalize_mad(energy_base)
    energy_term = -1.8 * norm_energy * energy_gate
    
    # Starvation-aware fairness: linear wait-time scaling with cap to prevent dominance
    max_wait = np.max(ready_wait_time + eps)
    capped_wait = np.minimum(ready_wait_time, max_wait * 0.5)
    fairness_raw = capped_wait / (max_wait + eps)
    norm_fairness = normalize_mad(fairness_raw)
    fairness_term = -0.3 * norm_fairness
    
    # Augmented penalty for slack violation amplified by uncertainty (from Parent 1 insight)
    slack_violation = np.maximum(-slack, 0.0)
    uncertainty_penalty = slack_violation * (1.0 + np.clip(uncertainty, 0.0, 10.0))
    norm_penalty = normalize_mad(uncertainty_penalty)
    penalty_term = 0.7 * norm_penalty  # Moderate weight: enforces DDL but avoids over-penalization
    
    # Combine terms with lexicographic priority: urgency dominates, then criticality, energy, fairness, penalty
    score = (
        4.0 * urgency_score +
        critical_term +
        energy_term +
        fairness_term +
        penalty_term
    )
    
    # Final sanitization: ensure finite, deterministic output with strict shape (N,)
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e8, 1e8)
    return score.reshape(-1)
