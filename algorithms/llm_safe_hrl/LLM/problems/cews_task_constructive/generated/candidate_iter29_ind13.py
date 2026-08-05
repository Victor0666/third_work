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
    # Sanitize all inputs: clip extremes, replace NaN/inf, ensure finite values
    eps = 1e-08
    def sanitize(x):
        x = np.asarray(x, dtype=float)
        x = np.clip(x, -1000000.0, 1000000.0)
        x = np.nan_to_num(x, nan=eps, posinf=1000000.0, neginf=-1000000.0)
        return x
    min_exec_time = sanitize(min_exec_time)
    min_comm_time = sanitize(min_comm_time)
    min_incremental_energy = sanitize(min_incremental_energy)
    slack = sanitize(slack)
    upward_rank = sanitize(upward_rank)
    remaining_work = sanitize(remaining_work)
    ready_wait_time = sanitize(ready_wait_time)
    uncertainty = sanitize(uncertainty)

    # Adaptive urgency tuning: higher tau when slack magnitude is large → smoother deadline pressure
    tau_urgency = 2.0 + np.clip(np.mean(np.abs(slack)) * 0.1, 0.5, 3.0)
    
    # Urgency: zero for non-critical tasks; linear penalty for negative slack (hard-deadline enforcement)
    urgency_raw = np.maximum(-slack, 0.0)
    
    # Latency pressure: critical-path density scaled by communication+exec time and uncertainty damping
    exec_comm_sum = min_exec_time + min_comm_time + eps
    cp_density = (upward_rank + eps) * (remaining_work + eps) / (exec_comm_sum + 1.0)
    uncertainty_damp = np.clip(1.0 - uncertainty, 0.05, 1.0)  # preserves signal even at high uncertainty
    latency_pressure = cp_density * uncertainty_damp
    
    # Safety-aware energy term: only rewards energy savings when slack is safe (soft-gated by safety factor)
    tau_safe = 5.0
    safety_factor = np.clip((slack + tau_safe) / (2.0 * tau_safe), 0.0, 1.0)
    energy_term = min_incremental_energy / (exec_comm_sum + 1.0) * safety_factor
    
    # Fairness: normalized wait time boosted by uncertainty when fairness risk is high (prevents starvation)
    wait_norm_base = ready_wait_time / (np.maximum(np.mean(ready_wait_time), eps) + eps)
    fairness_base = np.clip(wait_norm_base, 0.0, 5.0)
    fairness_boost = np.where(uncertainty > 0.4, uncertainty * 0.8, 0.0)  # stronger boost than v1
    fairness_term = fairness_base + fairness_boost
    
    # Robust min-max normalization: stable for N=1 and small N; avoids division-by-zero and outliers
    def robust_normalize(x):
        x = np.clip(x, -1000000.0, 1000000.0)
        if x.size == 1:
            return np.array([0.0])
        x_min, x_max = np.min(x), np.max(x)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x - x_min) / (x_max - x_min + eps)
    
    norm_urgency = robust_normalize(urgency_raw)
    norm_latency = robust_normalize(latency_pressure)
    norm_energy = robust_normalize(energy_term)
    norm_fair = robust_normalize(fairness_term)
    
    # Weighted score: urgency dominates (negative weight → smaller score = higher priority),
    # latency second, energy third; fairness has small positive weight to break ties without overriding deadlines
    score = -20.0 * norm_urgency - 14.0 * norm_latency - 6.0 * norm_energy + 0.5 * norm_fair
    
    # Hard-deadline violation boost: guarantees immediate selection for overdue tasks (not masked, not smoothed)
    violation_boost = np.where(slack < 0, -2000.0, 0.0)
    score = score + violation_boost
    
    # Final sanitization: ensure finite, bounded output with shape (N,)
    score = np.nan_to_num(score, nan=100000000.0, posinf=100000000.0, neginf=-100000000.0)
    score = np.clip(score, -100000000.0, 100000000.0)
    return score.astype(float).reshape(-1)
