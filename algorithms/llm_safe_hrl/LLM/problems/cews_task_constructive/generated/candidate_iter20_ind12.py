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
    v2 synthesis: Combines Parent 2's robust IQR normalization and soft slack gating
    with Parent 1's sharper urgency (tanh), risk-aware fairness, and critical-energy ratio.
    Key innovations:
    - Urgency: tanh(-slack/τ) with τ=0.5 → bounded, steeper near deadline, avoids arctan saturation
    - Critical-energy dominance: upward_rank / (min_incremental_energy + eps), soft-gated for slack >= -0.5s
    - Latency-efficiency term: (exec+comm)/energy only when slack > 0, promoting energy-latency tradeoff in safe region
    - Fairness: sqrt(wait) * exp(-uncertainty²) / (1 + |slack| + eps), always active but dampened under pressure
    - All terms normalized via adaptive IQR-based scheme with N=1 safety
    - Final score prioritizes DDL safety first (urgency dominates), then criticality-energy, then efficiency/fairness
    """
    eps = 1e-08
    tau_urgency = 0.5
    tau_violation = 0.5
    
    # Ensure finite, safe arrays
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    
    def normalize_adaptive(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        q25, q75 = np.percentile(x, [25, 75])
        iqr = q75 - q25
        if iqr > eps:
            center = np.median(x)
            scale = iqr + eps
        else:
            xmin, xmax = np.min(x), np.max(x)
            scale = xmax - xmin
            center = (xmax + xmin) / 2.0
            if scale < eps:
                scale = eps
        return (x - center) / (scale + eps)
    
    # Urgency: sharp, bounded, DDL-dominant (tanh)
    urgency_raw = np.tanh(-slack / tau_urgency)
    norm_urgency = normalize_adaptive(urgency_raw)
    urgency_term = -3.0 * norm_urgency  # Higher weight ensures deadline safety dominates
    
    # Critical-energy ratio: importance per marginal joule, soft-gated to allow energy optimization near deadline
    cer_base = upward_rank / (min_incremental_energy + eps)
    cer_masked = np.where(slack >= -tau_violation, cer_base, 0.0)
    norm_cer = normalize_adaptive(cer_masked)
    cer_term = -1.5 * norm_cer  # Strong negative weight → prioritize high-rank, low-energy tasks
    
    # Latency-efficiency: only active when slack > 0 (safe region), promotes energy-latency balance
    le_base = (min_exec_time + min_comm_time + eps) / (min_incremental_energy + eps)
    le_masked = np.where(slack > 0, le_base, 0.0)
    norm_le = normalize_adaptive(le_masked)
    le_term = 0.7 * norm_le  # Positive weight: lower efficiency → higher score → lower priority
    
    # Risk-aware fairness: prevents starvation while penalizing uncertainty; dampened by deadline pressure
    wait_safe = np.maximum(ready_wait_time, 0.0)
    sqrt_wait = np.sqrt(wait_safe + eps)
    risk_decay = np.exp(-np.square(uncertainty))
    fairness_raw = sqrt_wait * risk_decay
    fairness_scale = 1.0 + np.abs(slack) + eps
    fairness_scaled = fairness_raw / fairness_scale
    norm_fairness = normalize_adaptive(fairness_scaled)
    fairness_term = -0.3 * norm_fairness  # Negative weight → longer wait or lower uncertainty → higher priority
    
    # Assemble final score: urgency dominates, then critical-energy, then efficiency & fairness
    score = urgency_term + cer_term + le_term + fairness_term
    
    # Clamp to finite bounds
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    
    return score
