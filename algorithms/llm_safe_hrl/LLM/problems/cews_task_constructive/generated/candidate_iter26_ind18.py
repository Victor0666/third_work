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
    
    # Sanitize inputs: convert to float, replace NaN/inf with safe values
    def sanitize(x):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(x, nan=eps, posinf=1e6, neginf=-1e6)
    
    min_exec_time = sanitize(min_exec_time)
    min_comm_time = sanitize(min_comm_time)
    min_incremental_energy = sanitize(min_incremental_energy)
    slack = sanitize(slack)
    upward_rank = sanitize(upward_rank)
    remaining_work = sanitize(remaining_work)
    ready_wait_time = sanitize(ready_wait_time)
    uncertainty = sanitize(uncertainty)
    
    # Compute total latency (execution + communication), avoid division by zero
    total_latency = min_exec_time + min_comm_time + eps
    
    # Urgency modeling: linear penalty for negative slack, sigmoid for positive slack
    # Use adaptive thresholds based on slack distribution
    abs_slack = np.abs(slack)
    tau_tight = np.median(abs_slack[abs_slack < 3.0]) if np.any(abs_slack < 3.0) else 1.0
    tau_loose = np.median(abs_slack) + eps
    tau_urg = np.where(slack < 0, tau_tight, tau_loose)
    tau_urg = np.clip(tau_urg, 0.1, 100.0)
    
    # Linear urgency for overdue/near-deadline tasks (slack <= 0)
    urgency_linear = np.where(
        slack <= 0,
        1.0 - np.clip(slack / (tau_tight + eps), 0.0, 1.0),
        0.0
    )
    
    # Sigmoid urgency for slack > 0: rises smoothly near deadline margin
    urgency_sigmoid = np.where(
        slack > 0,
        1.0 / (1.0 + np.exp((slack - 2.0 * tau_loose) / (tau_loose + eps))),
        0.0
    )
    
    # Combine urgency signals; ensure monotonicity w.r.t. slack
    urgency_raw = np.maximum(urgency_linear, urgency_sigmoid)
    
    # Critical path importance: upward_rank * remaining_work, normalized robustly
    cp_score = upward_rank * remaining_work
    cp_med = np.median(cp_score) + eps
    cp_norm = cp_score / cp_med
    
    # Apply critical-path gating: prioritize high-uprank tasks when slack is non-negative
    # For overdue tasks (slack < 0), all tasks are equally critical for recovery
    uprank_thresh = np.percentile(upward_rank, 75) if upward_rank.size > 1 else np.mean(upward_rank)
    cp_mask = np.where(slack >= 0, (upward_rank >= uprank_thresh).astype(float), 1.0)
    cp_term = cp_norm * cp_mask
    
    # Energy efficiency term: energy per unit latency, gated by slack regime
    # Prioritize low-energy-per-latency only when slack is ample (reducing energy is secondary)
    energy_per_latency = min_incremental_energy / (total_latency + eps)
    energy_gate = np.where(
        slack > 5.0, 1.0,
        np.where(slack > 0, 0.3 + 0.7 * (slack / 5.0), 0.0)
    )
    
    # Uncertainty damping: reduce priority of high-uncertainty tasks when slack is tight
    # Stronger damping for negative/low slack; milder for large slack
    slack_margin_ratio = np.clip((slack + 1.0) / (np.abs(np.median(slack)) + 1.0 + eps), 0.0, 2.0)
    unc_damp = np.clip(1.0 - 0.4 * uncertainty * (1.0 / (slack_margin_ratio + eps)), 0.3, 1.0)
    
    energy_gated = energy_per_latency * energy_gate * unc_damp
    
    # Fairness boost: reward long-waiting tasks only when they are not overdue
    # Scale with both wait time and upward rank (higher importance tasks deserve faster service)
    wait_ratio = np.clip(ready_wait_time / (total_latency + eps), 0.0, 10.0)
    fairness_boost = np.where(
        slack >= 0,
        0.03 * wait_ratio * np.clip(upward_rank / (np.percentile(upward_rank, 90) + eps), 0.0, 1.0),
        0.0
    )
    
    # Final scaled components: normalize each to [0, 1] using robust percentiles
    def scale_to_unit(x):
        if x.size == 1:
            return np.array([0.5], dtype=float)
        p95 = np.percentile(x, 95) + eps
        scaled = np.clip(x / p95, 0.0, 1.0)
        return scaled
    
    urgency_scaled = scale_to_unit(urgency_raw)
    cp_scaled = scale_to_unit(cp_term)
    energy_scaled = scale_to_unit(energy_gated)
    
    # Weighted combination: urgency dominates, CP second, energy third
    # Fairness boost added directly to urgency to avoid diluting core objectives
    score = (
        0.55 * urgency_scaled +
        0.30 * cp_scaled +
        0.15 * energy_scaled +
        fairness_boost
    )
    
    # Final sanitization: ensure finite output, correct shape
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)
    
    # Enforce shape (N,) — reshape ensures column vector never returned
    return score.reshape(-1)
