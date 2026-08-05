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
    v2: Deadline-first, numerically stable, starvation-robust, and risk-aware priority.
    
    Key improvements:
    - Combines Parent 2's piecewise-linear urgency (strict DDL enforcement) with Parent 1's slack-pressure amplified critical-path term
    - Uses robust IQR-based normalization (Parent 2) but adds MAD fallback for degenerate cases (Parent 1)
    - Integrates Parent 1's *slack-gated energy efficiency ratio* (EER) with Parent 2's sigmoidal gating for smoother energy tradeoff
    - Adopts Parent 2's always-active wait_ratio fairness, enhanced with linear boost under slack abundance (Parent 1)
    - Applies uncertainty damping only on latency-sensitive tasks (tight feasible window), per Parent 1's insight
    - All operations are finite, deterministic, and sanitize NaN/inf/zero at every stage.
    """
    eps = 1e-08
    # Safe casting and nan/inf handling
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=0.0, posinf=eps, neginf=eps)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=0.0, posinf=eps, neginf=eps)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=eps, neginf=eps)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=eps, neginf=-eps)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=0.0, posinf=eps, neginf=0.0)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=0.0, posinf=eps, neginf=0.0)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=eps, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=0.0, posinf=eps, neginf=0.0)
    
    N = len(slack)
    if N == 0:
        return np.array([], dtype=float)
    
    def robust_normalize(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if N == 1:
            return np.array([0.0])
        # Prefer IQR; fallback to MAD if IQR is zero or degenerate
        q25, q50, q75 = np.percentile(x, [25, 50, 75], axis=0, keepdims=False)
        iqr = q75 - q25
        if iqr > eps:
            scale = iqr
        else:
            # MAD fallback
            abs_dev = np.abs(x - q50)
            mad = np.median(abs_dev)
            scale = mad if mad > eps else np.max(np.abs(x - q50)) if np.any(x != q50) else eps
        return (x - q50) / (scale + eps)
    
    # === Urgency: piecewise-linear risk (Parent 2) — strict & differentiable ===
    urgency_raw = np.where(slack < 0, -slack * 2.0, slack * 0.2)
    norm_urgency = robust_normalize(urgency_raw)
    urgency_term = -3.5 * norm_urgency
    
    # === Critical-path pressure: upward_rank × slack-aware amplification (Parent 1 + 2) ===
    positive_slack_mask = slack > 0
    median_pos_slack = np.median(slack[positive_slack_mask]) if np.any(positive_slack_mask) else 1.0
    slack_pressure = np.where(slack <= 0, 1.0 + (-slack) / (median_pos_slack + eps), 1.0)
    exec_effort = np.maximum(min_exec_time, eps)
    critical_density = upward_rank * (remaining_work / exec_effort)
    cp_pressure_base = critical_density * slack_pressure
    norm_cp = robust_normalize(cp_pressure_base)
    critical_term = -1.4 * norm_cp
    
    # === Energy efficiency ratio: EER = 1/(energy * (1 + |slack|)) gated by sigmoid (Parent 1 + 2) ===
    eer_base = 1.0 / (min_incremental_energy * (1.0 + np.abs(slack) + eps) + eps)
    slack_gate_sigmoid = 0.1 + 0.9 / (1.0 + np.exp(-slack / 1.0))  # smooth gate
    eer_gated = eer_base * slack_gate_sigmoid
    norm_eer = robust_normalize(eer_gated)
    efficiency_term = -0.95 * norm_eer
    
    # === Latency penalty with uncertainty damping only in tight feasible region (Parent 1) ===
    base_latency = min_exec_time + min_comm_time + eps
    tight_feasible_mask = (slack > 0) & (slack <= median_pos_slack)
    unc_q75 = np.percentile(uncertainty, 75) + eps
    uncertainty_factor = np.where(tight_feasible_mask, 
                                  1.0 + np.clip(uncertainty / unc_q75, 0.0, 2.0), 
                                  1.0)
    latency_penalty = base_latency * uncertainty_factor
    norm_latency = robust_normalize(latency_penalty)
    latency_term = 0.65 * norm_latency
    
    # === Fairness: always-active wait_ratio + linear wait-boost under slack abundance (Parent 2 + 1) ===
    duration_estimate = base_latency
    wait_ratio = np.clip(ready_wait_time / (duration_estimate + eps), 0.0, 10.0)
    # Linear boost when slack is abundant (exceeds median positive slack)
    wait_boost = np.where(slack > median_pos_slack, 
                          ready_wait_time / (slack + eps), 
                          0.0)
    combined_wait = wait_ratio + wait_boost
    norm_wait = robust_normalize(combined_wait)
    fairness_term = -0.42 * np.clip(norm_wait, -1.25, 1.75)
    
    # === Aggregate score ===
    score = urgency_term + critical_term + efficiency_term + latency_term + fairness_term
    
    # Final sanitization: ensure finite, deterministic output
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    
    return score
