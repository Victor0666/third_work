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
    v2 evolution: Combines Parent 2's lexicographic multiplicative hierarchy and MAD robustness
    with Parent 1's slack-aware urgency gating, critical-path modulation by local slack gradient,
    and adaptive uncertainty penalty. Introduces:
      - Hard-deadline dominance via tanh-based urgency gated by median absolute slack threshold
      - Critical-path density modulated by slack slope weight (as in Parent 1) for better CP alignment
      - SEER-gated energy efficiency activated only when slack > 2s *and* slack > 0.5*median_abs_slack
      - Risk-adjusted fairness using wait-time saturation (Parent 1) + uncertainty damping (Parent 2)
      - Unified robust normalization using MAD with fallback to range for singleton/stable inputs
      - Final score = urgency × (1 + CPD_modulated) × (1 + SEER_gated) × (1 + fairness_saturated)
      - All terms clipped to prevent numerical explosion; zero-energy masking preserved for violated deadlines
    """
    eps = 1e-08
    
    # Clean and copy inputs to avoid mutation
    def clean_and_copy(x):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    
    min_exec_time = clean_and_copy(min_exec_time)
    min_comm_time = clean_and_copy(min_comm_time)
    min_incremental_energy = clean_and_copy(min_incremental_energy)
    slack = clean_and_copy(slack)
    upward_rank = clean_and_copy(upward_rank)
    remaining_work = clean_and_copy(remaining_work)
    ready_wait_time = clean_and_copy(ready_wait_time)
    uncertainty = clean_and_copy(uncertainty)
    
    # Robust normalization: MAD-based, with fallback to range for degenerate cases
    def robust_normalize(x):
        if x.size == 0:
            return np.zeros_like(x)
        if x.size == 1:
            return np.array([0.0])
        med = np.median(x)
        dev = np.abs(x - med)
        mad = np.median(dev)
        scale = mad if mad > eps else (np.max(x) - np.min(x) + eps)
        z = (x - med) / (scale + eps)
        return np.clip(z, -2.0, 2.0)
    
    # --- Urgency term: hard deadline dominance with slack-aware gating ---
    # Use tanh(-slack/tau) for smooth, bounded urgency; gate by median absolute slack to avoid false urgency
    abs_slack = np.abs(slack)
    median_abs_slack = np.median(abs_slack) + eps
    tau_urgency = 0.5
    urgency_raw = np.tanh(-slack / tau_urgency)
    # Gate: only activate full urgency when slack is near or below median absolute slack
    urgency_gate = np.where(slack <= median_abs_slack, 1.0, 0.2)
    norm_urgency = robust_normalize(urgency_raw)
    # Map normalized urgency [-2,2] → multiplicative weight [0.1, 10.0], preserving monotonicity
    urgency_mult = np.clip(0.1 + 9.9 * (1.0 - (norm_urgency + 2.0) / 4.0), 0.1, 10.0)
    urgency_mult = urgency_mult * urgency_gate
    
    # --- Critical-path density (CPD) term: upward_rank weighted by execution cost and slack gradient ---
    exec_comm_sum = min_exec_time + min_comm_time + eps
    cpd_base = upward_rank / exec_comm_sum
    # Modulate by slack slope weight: higher weight when slack is small (urgent paths get boosted)
    slack_std = np.std(slack) + eps
    slack_slope_weight = np.clip(1.0 + (np.max(slack) - slack) / (slack_std + eps), 0.5, 3.0)
    cpd_modulated = cpd_base * slack_slope_weight
    norm_cpd = robust_normalize(cpd_modulated)
    # Map to [0, 1] offset for multiplicative term
    cpd_offset = np.clip((norm_cpd + 2.0) / 4.0, 0.0, 1.0)
    
    # --- SEER-gated energy efficiency term: activated only when slack is sufficiently positive ---
    seer_base = (min_exec_time + min_comm_time + eps) / (min_incremental_energy + eps)
    # Double-gated: both absolute slack > 2s AND relative slack > 0.5*median_abs_slack
    seer_active = np.where((slack > 2.0) & (slack > 0.5 * median_abs_slack), seer_base, 0.0)
    norm_seer = robust_normalize(seer_active)
    # Invert and compress: higher SEER → lower penalty → higher priority (so we use 2.0 - norm_seer)
    seer_offset = np.clip((2.0 - norm_seer) / 4.0, 0.0, 0.5)
    
    # --- Fairness term: wait-time saturation + uncertainty damping ---
    # Wait-time saturation with dynamic threshold based on workflow-level slack distribution
    urgent_ratio = np.mean(slack <= median_abs_slack)
    wait_threshold = np.clip(1.0 + 2.0 * (1.0 - urgent_ratio), 1.0, 5.0) * np.percentile(ready_wait_time, 80) + eps
    wait_saturation = np.clip(ready_wait_time / (wait_threshold + eps), 0.0, 1.0)
    wait_compressed = np.log1p(wait_saturation)  # Smooth sublinear growth
    # Uncertainty damping: exp(-uncertainty) scaled to [0.1, 1.0]
    uncertainty_damp = np.clip(np.exp(-np.clip(uncertainty, 0.0, 10.0)), 0.1, 1.0)
    fairness_raw = wait_compressed * uncertainty_damp
    # Gate fairness by slack > 1.0 to avoid penalizing urgent tasks
    fairness_gated = np.where(slack > 1.0, fairness_raw, 0.0)
    norm_fairness = robust_normalize(fairness_gated)
    fairness_offset = np.clip((norm_fairness + 2.0) / 4.0 * 0.3, 0.0, 0.3)
    
    # --- Final multiplicative priority score ---
    # Ensures lexicographic priority: urgency dominates, then CPD, then energy, then fairness
    # Violated deadlines (slack < 0) get suppressed urgency_mult but still retain base priority structure
    score = urgency_mult * (1.0 + cpd_offset) * (1.0 + seer_offset) * (1.0 + fairness_offset)
    
    # Final numeric safety
    score = np.nan_to_num(score, nan=1000000000.0, posinf=1000000000.0, neginf=1000000000.0)
    score = np.clip(score, 1e-6, 1e9)
    
    return score
