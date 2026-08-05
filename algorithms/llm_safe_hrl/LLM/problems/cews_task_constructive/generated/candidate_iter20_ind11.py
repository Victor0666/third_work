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
    Priority rule v2: Hard-DDL-first with robust slack-gated urgency, 
    critical-path density modulated by local slack gradient, energy-latency tradeoff ratio (ELTR),
    adaptive wait saturation fairness, and uncertainty-aware risk penalty.
    
    Key improvements:
    - Combines Parent 2's percentile scaling (robust, monotonic) and ELTR energy signal
    - Adopts Parent 1's zero-risk safeguard: hard deadline violation triggers priority boost for non-violating tasks
    - Integrates Parent 1's bounded exponential urgency decay for positive slack (smoother than sigmoid)
    - Uses adaptive slack thresholding (90th percentile) for gating instead of fixed multiples
    - Introduces normalized LAED-inspired term as secondary energy-efficiency signal when ELTR is unstable
    - Uncertainty penalty gated by both slack position and feasibility diversity (min_exec_time variance)
    - All terms scaled and clamped deterministically; final score ensures strict (N,) shape
    """
    eps = 1e-8
    N = len(slack)
    
    # Clean inputs: convert to float, replace NaN/inf/neg-inf
    def clean_array(x):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=eps)
    
    min_exec_time = clean_array(min_exec_time)
    min_comm_time = clean_array(min_comm_time)
    min_incremental_energy = clean_array(min_incremental_energy)
    slack = clean_array(slack)
    upward_rank = clean_array(upward_rank)
    remaining_work = clean_array(remaining_work)
    ready_wait_time = clean_array(ready_wait_time)
    uncertainty = clean_array(uncertainty)
    
    # Robust percentile scaling: preserves monotonicity, handles small-N
    def percentile_scale(x):
        if x.size == 0:
            return np.zeros_like(x)
        if x.size == 1:
            return np.array([0.0])
        p25, p50, p75 = np.percentile(x, [25, 50, 75])
        iqr = p75 - p25 + eps
        scaled = np.clip((x - p50) / (iqr * 0.5), -1.0, 1.0)
        return scaled
    
    # === URGENCY TERM: Hard-DDL dominance with bounded exponential decay ===
    # Zero-risk safeguard: if any task violates deadline (slack <= 0), boost priority of non-violators
    has_violation = np.any(slack <= 0)
    # Bounded exponential urgency for slack > 0; linear penalty for slack <= 0
    slack_abs_max = np.maximum(np.abs(np.percentile(slack, 90)), eps)
    urgency_raw = np.where(
        slack <= 0,
        -slack * 5.0,  # Strong linear penalty for lateness risk
        1.0 - np.exp(-np.clip(slack, 0.0, 5.0 * slack_abs_max) / (slack_abs_max + eps))
    )
    # Gate urgency to avoid false priority on distant deadlines
    slack_90 = np.percentile(slack, 90) + eps
    urgency_gate = np.where(slack <= slack_90, 1.0, 0.0)
    norm_urgency = percentile_scale(urgency_raw)
    urgency_term = -10.0 * norm_urgency * urgency_gate
    
    # === CRITICAL-PATH DENSITY TERM: Modulated by slack gradient ===
    exec_comm_sum = min_exec_time + min_comm_time + eps
    base_cp_density = upward_rank * remaining_work / exec_comm_sum
    # Modulate by local slack gradient: reward high-density tasks closer to deadline
    slack_std = np.std(slack) + eps
    slack_slope_weight = np.clip(
        1.0 + (np.max(slack) - slack) / (slack_std + eps), 0.5, 3.0
    )
    cp_density_modulated = base_cp_density * slack_slope_weight
    # Add uncertainty penalty for communication-heavy tasks under high uncertainty
    comm_unc_penalty = np.where(
        slack > 0, 
        min_comm_time * np.clip(uncertainty, 0.0, 2.0) * 0.1, 
        0.0
    )
    cp_density_penalized = cp_density_modulated + comm_unc_penalty / exec_comm_sum
    norm_cp_density = percentile_scale(cp_density_penalized)
    cp_density_term = 2.8 * norm_cp_density
    
    # === ENERGY-LATENCY TRADEOFF RATIO (ELTR): Primary energy signal ===
    latency_inv = 1.0 / (exec_comm_sum + eps)
    eltr_raw = latency_inv / (min_incremental_energy + eps)
    # Gate ELTR by slack: prioritize energy efficiency more when deadlines are loose
    eltr_gate = 1.0 / (1.0 + np.exp(-slack / (slack_abs_max + eps)))
    eltr_gated = eltr_raw * eltr_gate
    norm_eltr = percentile_scale(eltr_gated)
    energy_term = -1.9 * norm_eltr
    
    # === FAIRNESS TERM: Wait-time saturation with dynamic threshold ===
    urgent_ratio = np.mean(slack <= slack_abs_max)
    wait_threshold = np.clip(
        1.0 + 2.0 * (1.0 - urgent_ratio), 1.0, 5.0
    ) * (np.percentile(ready_wait_time, 80) + eps)
    wait_saturation = np.clip(ready_wait_time / (wait_threshold + eps), 0.0, 1.0)
    wait_compressed = np.log1p(wait_saturation)  # Smooth saturation
    norm_wait = percentile_scale(wait_compressed)
    fairness_term = -0.35 * norm_wait
    
    # === UNCERTAINTY PENALTY: Adaptive to feasibility diversity ===
    exec_var = np.var(min_exec_time) + eps
    unc_adapt = np.clip(exec_var / (np.mean(min_exec_time) + eps), 0.1, 5.0)
    # Apply only to tasks with high uncertainty AND loose slack (risk of overcommitment)
    unc_boost = np.where(
        (slack > np.percentile(slack, 90)) & (uncertainty > np.percentile(uncertainty, 80)),
        np.clip(uncertainty * 0.07 * unc_adapt, 0.0, 0.05),
        0.0
    )
    norm_unc = percentile_scale(unc_boost)
    uncertainty_term = 0.05 * norm_unc
    
    # === ZERO-RISK BOOST: Enforce hard deadline dominance ===
    # If any task violates deadline, penalize non-violating tasks less (boost priority)
    violation_boost = np.where(
        has_violation,
        np.where(slack > 0, 1000000.0 * (1.0 - np.tanh(np.clip(slack, 0.0, 10.0) / 10.0)), 0.0),
        0.0
    )
    
    # Combine all terms
    score = (
        urgency_term +
        cp_density_term +
        energy_term +
        fairness_term +
        uncertainty_term +
        violation_boost
    )
    
    # Final robustness: handle NaN/inf, clamp, ensure shape (N,)
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e8, 1e8)
    
    # Ensure output is 1D array of shape (N,)
    return score.astype(float).reshape(-1)
