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
    
    # Sanitize all inputs: convert to float, replace NaN/inf with safe values
    def sanitize(x):
        x = np.asarray(x, dtype=float)
        x = np.nan_to_num(x, nan=eps, posinf=1e6, neginf=-1e6)
        return np.clip(x, -1e6, 1e6)
    
    min_exec_time = sanitize(min_exec_time)
    min_comm_time = sanitize(min_comm_time)
    min_incremental_energy = sanitize(min_incremental_energy)
    slack = sanitize(slack)
    upward_rank = sanitize(upward_rank)
    remaining_work = sanitize(remaining_work)
    ready_wait_time = sanitize(ready_wait_time)
    uncertainty = sanitize(uncertainty)
    
    # Robust slack: proactive deadline margin accounting for uncertainty
    robust_slack = slack - 2.0 * uncertainty
    robust_slack = np.clip(robust_slack, -1e6, 1e6)
    
    # --- Urgency: piecewise-linear + sigmoid with adaptive thresholds ---
    abs_robust_slack = np.abs(robust_slack)
    med_slack = np.median(abs_robust_slack) + eps
    q1 = np.percentile(abs_robust_slack, 25) if abs_robust_slack.size > 1 else med_slack * 0.5
    q3 = np.percentile(abs_robust_slack, 75) if abs_robust_slack.size > 1 else med_slack * 1.5
    iqr = q3 - q1 + eps
    tau_tight = np.clip(med_slack * (0.4 + 0.6 * (iqr / (med_slack + eps))), 0.05, 10.0)
    tau_loose = np.clip(med_slack + 1.5 * iqr, 0.5, 200.0)
    
    # Linear urgency for violating/near-violating tasks (hard priority boost)
    urgency_linear = np.where(
        robust_slack <= 0,
        1.0 - np.clip(robust_slack / (tau_tight + eps), 0.0, 1.0),
        0.0
    )
    
    # Sigmoid urgency for non-violating tasks (smooth decay beyond safe margin)
    urgency_sigmoid = np.where(
        robust_slack > 0,
        1.0 / (1.0 + np.exp((robust_slack - tau_loose) / (iqr + eps))),
        0.0
    )
    
    urgency_raw = np.clip(np.maximum(urgency_linear, urgency_sigmoid), 0.0, 1.0)
    
    # --- Critical path pressure: upward_rank weighted by work and latency ---
    total_latency = min_exec_time + min_comm_time + eps
    cp_score = upward_rank * remaining_work * (1.0 + 0.5 * uncertainty)
    
    # IQR-based normalization for robustness
    cp_med = np.median(cp_score) + eps
    if cp_score.size > 1:
        cp_q1 = np.percentile(cp_score, 25)
        cp_q3 = np.percentile(cp_score, 75)
        cp_iqr = cp_q3 - cp_q1 + eps
    else:
        cp_iqr = cp_med * 0.1 + eps
    cp_norm = (cp_score - cp_med) / (cp_iqr + eps)
    cp_scaled = np.clip(cp_norm, -3.0, 3.0) / 6.0 + 0.5
    
    # Gate critical path term by slack to avoid over-prioritizing non-urgent tasks
    cp_gate = np.clip(1.0 + robust_slack / (tau_tight + eps), 0.0, 1.0)
    cp_term = cp_scaled * cp_gate
    
    # --- Energy efficiency: marginal energy per unit work (lower is better) ---
    energy_per_work = min_incremental_energy / (remaining_work + eps)
    
    # Normalize energy term with IQR; invert so lower energy → lower score
    energy_med = np.median(energy_per_work) + eps
    if energy_per_work.size > 1:
        energy_q1 = np.percentile(energy_per_work, 25)
        energy_q3 = np.percentile(energy_per_work, 75)
        energy_iqr = energy_q3 - energy_q1 + eps
    else:
        energy_iqr = energy_med * 0.1 + eps
    energy_norm = (energy_per_work - energy_med) / (energy_iqr + eps)
    energy_scaled = np.clip(energy_norm, -3.0, 3.0) / 6.0 + 0.5
    
    # Energy gating: only penalize energy when slack is healthy (prioritize DDL first)
    energy_gate = np.where(
        robust_slack > 0,
        np.clip(1.0 - (tau_loose - robust_slack) / (tau_loose + eps), 0.0, 1.0),
        0.0
    )
    # Apply uncertainty damping: higher uncertainty reduces energy penalty weight
    unc_damp = np.clip(1.0 - 0.4 * uncertainty, 0.2, 1.0)
    energy_gated = energy_scaled * energy_gate * unc_damp
    
    # --- Fairness: age-based boosting for long-waiting tasks, gated by slack ---
    wait_ratio = np.clip(ready_wait_time / (total_latency + eps), 0.0, 50.0)
    fairness_boost = np.where(
        robust_slack >= 0,
        0.1 * wait_ratio * np.clip(1.0 + 0.1 * upward_rank, 1.0, 2.5),
        0.0
    )
    
    # --- Final linear combination with violation floor ---
    # Violating tasks get absolute top priority (score = 0.0)
    violation_mask = (slack < 0).astype(bool)
    
    # Weighted sum: urgency dominates (0.55), then CP (0.25), energy (0.15), fairness (0.05)
    score = (
        0.55 * urgency_raw +
        0.25 * cp_term +
        0.15 * energy_gated +
        0.05 * fairness_boost
    )
    
    # Apply floor for violations — ensures hard DDL adherence
    score = np.where(violation_mask, 0.0, score)
    
    # Final sanitization: ensure finite, clipped, correct shape
    score = np.nan_to_num(score, nan=1e6, posinf=1e6, neginf=0.0)
    score = np.clip(score, 0.0, 1e6)
    score = np.asarray(score, dtype=float).reshape(-1)
    
    return score
