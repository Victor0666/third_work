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
    eps = 1e-08
    tau = 1.0
    
    def clean(x):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Clean all inputs without in-place modification
    min_exec_time = clean(min_exec_time)
    min_comm_time = clean(min_comm_time)
    min_incremental_energy = clean(min_incremental_energy)
    slack = clean(slack)
    upward_rank = clean(upward_rank)
    remaining_work = clean(remaining_work)
    ready_wait_time = clean(ready_wait_time)
    uncertainty = clean(uncertainty)
    
    # --- Dynamic weight scaling based on global deadline pressure ---
    # Compute system-wide urgency: fraction of ready tasks already at risk (slack < 0)
    deadline_pressure = np.mean((slack < 0.0).astype(float))
    # Scale urgency weight inversely with pressure to avoid over-aggression when many tasks are late
    urgency_weight = 4.0 * (1.0 + 0.5 * deadline_pressure)  # increases slightly under high pressure
    # Scale energy weight positively with slack headroom to promote efficiency only when safe
    energy_weight_base = 1.8
    energy_weight = energy_weight_base * np.clip(np.mean(np.maximum(slack, 0.0)) / (np.max(np.abs(slack)) + eps), 0.0, 1.0)
    
    # --- Risk-adjusted urgency (bounded, smooth, uncertainty-aware) ---
    risk_adjusted_slack = slack - 2.0 * uncertainty
    urgency_raw = np.tanh(risk_adjusted_slack / (tau + eps))
    # Invert: tanh(-∞→∞) ∈ [-1,1] → (1-urgency_raw)/2 ∈ [0,1], smaller = more urgent
    urgency_score = (1.0 - urgency_raw) / 2.0
    
    # --- Robust MAD-based normalization with zero-variance safety ---
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
    
    # --- Criticality term: active only when slack < 0, weighted by path importance and work ---
    critical_mask = (slack < 0.0).astype(float)
    norm_remaining_work = remaining_work / (np.max(remaining_work + eps) + eps)
    critical_base = upward_rank * norm_remaining_work
    norm_critical = normalize_mad(critical_base)
    critical_term = 2.5 * norm_critical * critical_mask
    
    # --- Energy-efficiency term: activated only when slack > 0, scaled by headroom ---
    total_latency = min_exec_time + min_comm_time + eps
    seer_ratio = min_incremental_energy / total_latency
    # Invert SEER: higher energy-per-work → lower priority → larger score → use negative sign
    energy_base = 1.0 / (seer_ratio + eps)
    energy_gate = (slack > 0.0).astype(float)
    norm_energy = normalize_mad(energy_base)
    energy_term = -energy_weight * norm_energy * energy_gate
    
    # --- Fairness term: linear wait-time with adaptive cap to prevent dominance ---
    max_wait = np.max(ready_wait_time + eps)
    # Cap at 30% of max wait to limit starvation penalty in long simulations
    capped_wait = np.minimum(ready_wait_time, max_wait * 0.3)
    fairness_raw = capped_wait / (max_wait + eps)
    norm_fairness = normalize_mad(fairness_raw)
    fairness_term = -0.3 * norm_fairness
    
    # --- Uncertainty-penalized lateness: explicit penalty for violated slack, scaled by uncertainty ---
    slack_violation = np.maximum(-slack, 0.0)
    # Penalize both magnitude and uncertainty amplification, but cap uncertainty effect
    uncertainty_factor = 1.0 + np.clip(uncertainty, 0.0, 5.0)
    penalty_base = slack_violation * uncertainty_factor
    norm_penalty = normalize_mad(penalty_base)
    penalty_term = 0.7 * norm_penalty
    
    # --- Combine with dynamically scaled weights ---
    score = (
        urgency_weight * urgency_score +
        critical_term +
        energy_term +
        fairness_term +
        penalty_term
    )
    
    # --- Final sanitization: ensure finite, deterministic, shape-(N,) output ---
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e8, 1e8)
    
    # Enforce shape (N,) — no scalars or column vectors
    return score.reshape(-1)
