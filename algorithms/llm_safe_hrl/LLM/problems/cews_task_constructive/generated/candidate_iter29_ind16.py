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
    
    # Sanitize inputs: ensure finite, replace NaN/inf with safe values
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
    
    # Adaptive deadline sensitivity via IQR-based tau estimation (Parent 2 strength)
    abs_slack = np.abs(slack)
    med_slack = np.median(abs_slack) + eps
    q1 = np.percentile(abs_slack, 25) if abs_slack.size > 1 else med_slack * 0.5
    q3 = np.percentile(abs_slack, 75) if abs_slack.size > 1 else med_slack * 1.5
    iqr = q3 - q1 + eps
    tau_tight = np.clip(med_slack * (0.5 + 0.5 * (iqr / (med_slack + eps))), 0.05, 10.0)
    tau_loose = np.clip(med_slack + iqr, 0.5, 200.0)
    
    # Urgency: smooth, differentiable gating (Parent 2) + risk-adjusted slack (Parent 1 inspired)
    # Introduce uncertainty-aware slack shift to proactively prioritize high-risk near-deadline tasks
    risk_adjusted_slack = slack - 1.5 * uncertainty  # bias toward earlier scheduling of uncertain tasks
    urgency_linear = np.where(risk_adjusted_slack <= 0, 
                             1.0 - np.clip(risk_adjusted_slack / (tau_tight + eps), 0.0, 1.0), 
                             0.0)
    urgency_sigmoid = np.where(risk_adjusted_slack > 0, 
                              1.0 / (1.0 + np.exp((risk_adjusted_slack - tau_loose) / (iqr + eps))), 
                              0.0)
    urgency_raw = np.clip(np.maximum(urgency_linear, urgency_sigmoid), 0.0, 1.0)
    
    # Criticality: HEFT-inspired but only activated when slack is tight (not just negative)
    # Use normalized critical path contribution, gated by slack safety margin (Parent 2 CP gate + Parent 1 critical masking insight)
    cp_score = upward_rank * remaining_work
    cp_med = np.median(cp_score) + eps
    cp_iqr = (np.percentile(cp_score, 75) - np.percentile(cp_score, 25) + eps) if cp_score.size > 1 else cp_med * 0.1
    cp_norm = (cp_score - cp_med) / (cp_iqr + eps)
    cp_scaled = np.clip(cp_norm, -3.0, 3.0) / 6.0 + 0.5
    # Gate criticality using *normalized slack distance* — stronger emphasis when slack < tau_tight
    cp_gate = np.clip(1.0 + (tau_tight - slack) / (tau_tight + eps), 0.0, 1.0)
    cp_term = cp_scaled * cp_gate
    
    # Energy term: activate only when sufficient slack remains and dampen for high uncertainty
    total_latency = min_exec_time + min_comm_time + eps
    energy_per_latency = min_incremental_energy / (total_latency + eps)
    # Energy optimization enabled only when slack > tau_loose (conservative safety margin)
    energy_gate = np.where(slack > tau_loose, 
                          np.clip((slack - tau_loose) / (tau_loose + eps), 0.0, 1.0), 
                          0.0)
    # Uncertainty-aware damping: higher uncertainty → lower priority for energy saving (avoid risky low-energy assignment)
    unc_damp = np.clip(1.0 - 0.4 * uncertainty, 0.2, 1.0)
    energy_gated = energy_per_latency * energy_gate * unc_damp
    energy_med = np.median(energy_gated) + eps
    energy_iqr = (np.percentile(energy_gated, 75) - np.percentile(energy_gated, 25) + eps) if energy_gated.size > 1 else energy_med * 0.1
    energy_scaled = np.clip((energy_gated - energy_med) / (energy_iqr + eps), -3.0, 3.0) / 6.0 + 0.5
    
    # Fairness: starvation mitigation with uncertainty amplification (Parent 1 insight) + latency-normalized wait time
    # Prioritize long-waiting tasks, especially under high uncertainty — prevents indefinite deferral
    wait_ratio = np.clip(ready_wait_time / (total_latency + eps), 0.0, 50.0)
    fairness_boost = wait_ratio * (1.0 + 0.2 * uncertainty)  # amplify wait priority for uncertain tasks
    # Only apply fairness boost when slack allows (don’t sacrifice DDL for fairness)
    fairness_gate = np.where(slack >= -tau_tight, 1.0, 0.0)
    fairness_term = fairness_boost * fairness_gate
    
    # Final weighted combination: urgency dominates, criticality secondary, energy tertiary, fairness auxiliary
    # Weights tuned for DDL-hard constraint: urgency gets largest share; fairness avoids livelock without compromising deadlines
    score = (
        0.65 * urgency_raw +
        0.20 * cp_term +
        0.10 * energy_scaled +
        0.05 * fairness_term
    )
    
    # Final sanitization: guarantee finite, bounded, shape-(N,) output
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e8, 1e8)
    return score.reshape(-1)
