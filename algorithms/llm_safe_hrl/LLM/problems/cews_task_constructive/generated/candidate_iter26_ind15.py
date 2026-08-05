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
    
    # Sanitize all inputs: ensure float, replace NaN/inf with safe values
    def sanitize(x):
        x = np.asarray(x, dtype=float)
        x = np.nan_to_num(x, nan=eps, posinf=1e6, neginf=-1e6)
        return x
    
    min_exec_time = sanitize(min_exec_time)
    min_comm_time = sanitize(min_comm_time)
    min_incremental_energy = sanitize(min_incremental_energy)
    slack = sanitize(slack)
    upward_rank = sanitize(upward_rank)
    remaining_work = sanitize(remaining_work)
    ready_wait_time = sanitize(ready_wait_time)
    uncertainty = sanitize(uncertainty)
    
    # --- Urgency: dual-mode handling of deadline pressure ---
    # For overdue/near-deadline tasks: linear urgency (hard constraint emphasis)
    tau_tight = np.clip(np.median(np.abs(slack[slack < 3.0])) if np.any(slack < 3.0) else 1.0, 0.1, 5.0)
    urgency_linear = np.where(slack <= 0,
                              1.0 - np.clip(slack / (tau_tight + eps), 0.0, 1.0),
                              0.0)
    
    # For ahead-of-deadline tasks: sigmoid soft urgency (prioritizes critical path within margin)
    tau_loose = np.clip(np.median(np.abs(slack)) + eps, 1.0, 100.0)
    urgency_sigmoid = np.where(slack > 0,
                               1.0 / (1.0 + np.exp((slack - 2.0 * tau_loose) / (tau_loose + eps))),
                               0.0)
    
    # Combine: max ensures hard-deadline urgency dominates; clamp to [0,1]
    urgency_raw = np.clip(np.maximum(urgency_linear, urgency_sigmoid), 0.0, 1.0)
    
    # --- Critical Path Importance (CP) term ---
    total_latency = min_exec_time + min_comm_time + eps
    cp_score = upward_rank * remaining_work  # HEFT-inspired criticality proxy
    
    # Normalize CP score relative to its own 95th percentile for stability
    cp_p95 = np.percentile(cp_score, 95) if cp_score.size > 1 else np.max(cp_score)
    cp_scaled = np.clip(cp_score / (cp_p95 + eps), 0.0, 1.0)
    
    # Gate CP contribution: full weight when slack >= 0 (no violation), reduced when tight
    cp_gate = np.where(slack >= 0, 1.0, np.clip(0.5 + 0.5 * (slack / (tau_tight + eps)), 0.0, 1.0))
    cp_term = cp_scaled * cp_gate
    
    # --- Energy Efficiency Term: reward low energy-per-latency only when slack allows ---
    energy_per_latency = min_incremental_energy / (total_latency + eps)
    
    # Energy gating: only activate strong energy preference when slack > 5s (safe margin)
    energy_gate = np.where(slack > 5.0, 1.0,
                          np.where(slack > 0.0, 0.3 + 0.7 * (slack / 5.0), 0.0))
    
    # Uncertainty damping: penalize high-uncertainty tasks unless slack is negative (urgent override)
    unc_damp = np.clip(1.0 - 0.4 * uncertainty, 0.3, 1.0)
    energy_gated = energy_per_latency * energy_gate * unc_damp
    
    # Normalize energy term using 95th percentile for outlier resilience
    energy_p95 = np.percentile(energy_gated, 95) if energy_gated.size > 1 else np.max(energy_gated)
    energy_scaled = np.clip(energy_gated / (energy_p95 + eps), 0.0, 1.0)
    
    # --- Fairness & Aging Boost: mitigate starvation without violating deadlines ---
    wait_ratio = np.clip(ready_wait_time / (total_latency + eps), 0.0, 10.0)
    # Only apply fairness boost when slack is non-negative (no risk of missing DDL)
    fairness_boost = np.where(slack >= 0,
                              0.03 * wait_ratio * np.clip(1.0 + 0.1 * upward_rank, 1.0, 2.0),
                              0.0)
    
    # --- Final weighted fusion: urgency dominates, CP guides structure, energy saves when safe ---
    # Weights sum to 1.0 and reflect objective priority: meet DDL first, then minimize energy
    score = (
        0.55 * urgency_raw +
        0.30 * cp_term +
        0.15 * energy_scaled +
        fairness_boost
    )
    
    # Final sanitization: ensure finite, bounded, shape-(N,) output
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)
    
    # Enforce shape (N,) — reshape to 1D column-safe vector
    return score.reshape(-1)
