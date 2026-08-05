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
    
    # Sanitize all inputs: ensure finite, non-NaN, bounded values
    def sanitize(x):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(x, nan=eps, posinf=1e6, neginf=1e-6)
    
    min_exec_time = sanitize(min_exec_time)
    min_comm_time = sanitize(min_comm_time)
    min_incremental_energy = sanitize(min_incremental_energy)
    slack = sanitize(slack)
    upward_rank = sanitize(upward_rank)
    remaining_work = sanitize(remaining_work)
    ready_wait_time = sanitize(ready_wait_time)
    uncertainty = sanitize(uncertainty)
    
    # Compute robust slack: penalize uncertainty strongly but clamp to avoid instability
    robust_slack = slack - 3.0 * uncertainty
    robust_slack = np.clip(robust_slack, -1e5, 1e5)
    
    # Lexicographic violation override: finite dominant penalty for deadline violations
    # Ensures violated tasks always get highest priority (lowest score) without -inf
    violation_penalty = np.where(slack < 0, -1e9, 0.0)
    
    # Urgency: arctan-based bounded urgency (-π/2 to π/2), mapped to [0.01, 10.0]
    # Uses robust_slack to avoid overreacting to noise; avoids tanh saturation
    urgency_input = robust_slack / (np.abs(robust_slack) + eps)
    urgency_arctan = np.arctan(urgency_input)
    urgency_score = 0.01 + 9.99 * (1.0 - (urgency_arctan + np.pi/2) / np.pi)
    
    # Critical path density: upward_rank * work / (exec+comm), normalized safely
    exec_comm_sum = min_exec_time + min_comm_time + eps
    cp_density = upward_rank * remaining_work / (exec_comm_sum + eps)
    cp_density = np.clip(cp_density, 1e-6, 1e6)
    
    # MAD normalization with fallback for small N: identity if size <= 2, else MAD
    def normalize_safe(x):
        x = np.clip(x, -1e6, 1e6)
        if x.size <= 2:
            return np.zeros_like(x)  # no meaningful scale; neutral offset
        median = np.median(x)
        mad = np.median(np.abs(x - median))
        if mad < eps:
            return np.zeros_like(x)
        return (x - median) / (mad + eps)
    
    norm_cp = normalize_safe(cp_density)
    # Sigmoid-compressed offset: maps [-inf,inf] → (0.5, 1.5) for stable multiplicative influence
    cp_offset = 0.5 + 1.0 / (1.0 + np.exp(-norm_cp))
    
    # SEER (Speed-Energy Efficiency Ratio): (exec+comm)/energy → higher is better → invert for priority
    seer = exec_comm_sum / (min_incremental_energy + eps)
    seer = np.clip(seer, 1e-6, 1e6)
    seer_arctan = np.arctan(-seer / 10.0)  # scaled to avoid flat regions
    seer_offset = 0.5 + 0.5 * (1.0 + seer_arctan / (np.pi/2))
    
    # Fairness: aging only when slack is robustly positive (feasibility-aware)
    fairness_raw = ready_wait_time / (np.abs(slack) + 1.0)
    fairness_gated = np.where(robust_slack > 0.0, fairness_raw, 0.0)
    norm_fair = normalize_safe(fairness_gated)
    fairness_offset = 0.5 + 0.5 / (1.0 + np.exp(-norm_fair))
    
    # Log-space composition: sum of logs → product in linear space, more numerically stable
    # Prevents vanishing/exploding gradients from multiplication of small/large offsets
    log_score = (
        np.log(urgency_score + eps) +
        np.log(cp_offset + eps) +
        np.log(seer_offset + eps) +
        np.log(fairness_offset + eps)
    )
    
    # Combine with finite violation penalty in linear space (lexicographic dominance)
    # Violation penalty dominates any log-score magnitude
    score = np.exp(log_score) + violation_penalty
    
    # Final sanitization: ensure finite, deterministic, shape-(N,) output
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)
    
    # Ensure shape is strictly (N,) — critical for single-task case
    return score.astype(float)
