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
    # Sanitize inputs: convert to float, replace NaN/inf with safe finite values
    eps = 1e-08
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

    # --- Deadline Hardening & Urgency Modeling ---
    # Robust slack: reduce effective slack under uncertainty (conservative deadline view)
    robust_slack = np.where(slack > 0, slack - uncertainty * 0.5, slack)
    
    # Piecewise urgency: monotonic, bounded [0,1], zero at slack=0, sharp rise near deadline
    # Linear ramp for slack <= 0 (violation zone), sigmoid for slack > 0 (risk zone)
    urgency_linear = np.clip(-robust_slack / (eps + 1.0), 0.0, 1.0)  # strong push when slack ≤ 0
    urgency_sigmoid = np.where(
        robust_slack > 0,
        1.0 / (1.0 + np.exp(-(robust_slack - 2.0) / (0.5 + eps))),  # smooth rise centered at slack=2s
        0.0
    )
    urgency_signal = np.maximum(urgency_linear, urgency_sigmoid)
    
    # Hard violation priority boost: tasks with robust_slack < -eps get fixed top priority
    violation_mask = robust_slack < -eps
    base_urgency = urgency_signal.copy()
    base_urgency = np.where(violation_mask, 1.0, base_urgency)

    # --- Latency-Criticality Term ---
    # Critical path pressure: importance × latency × urgency-scaling × uncertainty damping
    exec_comm_sum = min_exec_time + min_comm_time + eps
    latency_pressure = upward_rank * exec_comm_sum * (1.0 + base_urgency) * np.clip(1.0 - uncertainty, 0.2, 1.0)

    # --- Energy Efficiency Term ---
    # Penalize high energy per remaining work, but only when safety margin is low
    energy_per_work = min_incremental_energy / (remaining_work + eps)
    safety_margin = np.clip(robust_slack / (3.0 + eps), 0.0, 1.0)  # tau_safe = 3.0
    energy_term = energy_per_work * (1.0 - safety_margin) * (1.0 + base_urgency)

    # --- Critical Path Density (CPD) Term ---
    # Upward rank density: captures importance-per-latency; scaled by urgency & uncertainty
    cp_density = (upward_rank + eps) * (remaining_work + eps) / (exec_comm_sum + eps)
    cp_gated = cp_density * (1.0 + base_urgency) * np.clip(1.0 - uncertainty, 0.2, 1.0)

    # --- Fairness Term (Starvation Relief) ---
    # Only activated when task is safe (robust_slack > 3.0) AND uncertain (uncertainty > 0.2)
    fairness_base = ready_wait_time * (1.0 + upward_rank) / (exec_comm_sum + eps)
    fairness_boost = np.where(
        (robust_slack > 3.0) & (uncertainty > 0.2),
        uncertainty * 0.7 * fairness_base,
        0.0
    )
    fairness_term = fairness_base + fairness_boost

    # --- Robust Min-Max Normalization (per-term, stable for N=1) ---
    def robust_normalize(x):
        x = np.clip(x, -1e5, 1e5)
        if x.size == 1:
            return np.array([0.0])
        x_min, x_max = np.min(x), np.max(x)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x - x_min) / (x_max - x_min + eps)
    
    norm_urgency = robust_normalize(base_urgency)
    norm_latency = robust_normalize(latency_pressure)
    norm_energy = robust_normalize(energy_term)
    norm_cp = robust_normalize(cp_gated)
    norm_fair = robust_normalize(fairness_term)

    # --- Final Score: Deterministic, finite, deadline-first prioritization ---
    # Strong negative weights for urgency and latency → smaller score = higher priority
    # Positive weight for CP density reflects *beneficial* critical-path progression
    # Fairness has small negative weight (mild penalty to avoid over-prioritizing old safe tasks)
    score = (
        -30.0 * norm_urgency      # Dominant: strict deadline enforcement
        - 18.0 * norm_latency     # High penalty for latency-critical tasks under urgency
        - 10.0 * norm_energy      # Energy efficiency matters most when deadlines are tight
        +  2.0 * norm_cp          # Reward progress on critical path (positive incentive)
        -  0.8 * norm_fair        # Mild fairness correction only when safe & uncertain
    )

    # Hard priority override: DDL-violating tasks get lowest possible score
    if np.any(violation_mask):
        non_violating_scores = score[~violation_mask] if np.any(~violation_mask) else score
        base_min = np.min(non_violating_scores)
        score = np.where(violation_mask, base_min - 1e9, score)

    # Final sanitization: ensure finite, deterministic output of shape (N,)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)
    return score.reshape(-1)
