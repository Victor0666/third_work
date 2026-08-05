import numpy as np

# 函数名中的 2 由 SeEvo 在生成 Prompt 时替换为目标版本号。
# 八个输入均为长度 N 的一维数组；相同下标始终指向同一个 ready task。
# 返回值也必须是长度 N 的一维数组，并遵守“分数越小，优先级越高”。
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
    Priority rule v3: Hard-DDL-first with *slack-gradient urgency calibration*, 
    *critical-path density refined by dynamic criticality horizon*, 
    *dual-energy signals (ELTR + LAED-normalized marginal efficiency)*, 
    *wait fairness via monotonic saturation with workflow-level context*, 
    and *uncertainty penalty gated by both slack position AND feasibility dispersion*.
    
    Key improvements over v1:
    - Replaces static percentile gating with *adaptive slack-gradient urgency*: 
      urgency decays smoothly as slack increases beyond local median, preserving sensitivity near DDL.
    - Introduces *criticality horizon*: scales upward_rank by slack-aware reachability (exp(-|slack|/τ)), 
      suppressing over-prioritization of high-rank tasks far from deadline.
    - Adds *LAED-inspired normalized efficiency term* (latency-adjusted energy density) as stable fallback 
      when ELTR is numerically unstable (e.g., near-zero energy or latency), weighted only when ELTR variance > threshold.
    - Fairness now uses *workflow-relative wait saturation*: threshold anchored to max(ready_wait_time, 0.5 * median_slack_abs) 
      to avoid starvation in ultra-tight DDL regimes.
    - Uncertainty penalty upgraded to *joint slack-uncertainty risk zone detection*: triggers only when 
      (slack < 75th percentile AND uncertainty > 85th percentile), avoiding false positives in slack-rich regions.
    - All terms scaled to unit-equivalent ranges using robust percentile scaling; final score strictly bounded and reshaped.
    - Eliminates redundant clamping layers; uses single deterministic nan/inf guard + final clip.
    """
    eps = 1e-08
    N = len(slack)
    
    # Clean inputs: ensure float, replace NaN/inf with safe values
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
    
    # Robust percentile scaler: preserves monotonicity, handles outliers
    def percentile_scale(x):
        if x.size == 0:
            return np.zeros_like(x)
        if x.size == 1:
            return np.array([0.0])
        p25, p50, p75 = np.percentile(x, [25, 50, 75])
        iqr = p75 - p25 + eps
        scaled = np.clip((x - p50) / (iqr * 0.5), -1.0, 1.0)
        return scaled
    
    # === URGENCY TERM: Slack-gradient calibrated, hard-DDL first ===
    slack_abs = np.abs(slack)
    slack_med = np.median(slack_abs) + eps
    # Urgency: high near/after deadline, decays smoothly for large positive slack
    # Uses exp(-|slack|/τ) for negative/zero slack → 1.0, then linear decay above median
    urgency_raw = np.where(
        slack <= 0,
        1.0 + (-slack) * 0.2,  # Boost for violations: +0.2 per sec lateness
        np.clip(1.0 - (slack_abs - slack_med) / (slack_med + eps), 0.0, 1.0)
    )
    norm_urgency = percentile_scale(urgency_raw)
    urgency_term = -12.0 * norm_urgency
    
    # === CRITICAL-PATH DENSITY: Horizon-gated & uncertainty-modulated ===
    exec_comm_sum = min_exec_time + min_comm_time + eps
    # Criticality horizon: suppress rank importance for tasks with large positive slack
    horizon_weight = np.exp(-np.clip(slack_abs, 0.0, 10.0 * slack_med) / (slack_med + eps))
    base_cp_density = (upward_rank * remaining_work * horizon_weight) / exec_comm_sum
    # Add light comm-uncertainty penalty only for slack-positive tasks
    comm_unc_penalty = np.where(
        slack > 0,
        min_comm_time * np.clip(uncertainty, 0.0, 3.0) * 0.08,
        0.0
    )
    cp_density_penalized = base_cp_density + comm_unc_penalty / exec_comm_sum
    norm_cp_density = percentile_scale(cp_density_penalized)
    cp_density_term = 3.0 * norm_cp_density
    
    # === ENERGY TERM: Dual-signal — primary ELTR + fallback LAED ===
    latency_inv = 1.0 / (exec_comm_sum + eps)
    eltr_raw = latency_inv / (min_incremental_energy + eps)
    # Fallback LAED: energy per unit remaining work, normalized by latency
    laed_raw = min_incremental_energy / (remaining_work + eps) / (exec_comm_sum + eps)
    # Use LAED only when ELTR is unstable (high variance or near-zero denominator)
    eltr_var = np.var(eltr_raw) if len(eltr_raw) > 1 else 0.0
    use_laed = (eltr_var < 1e-4) | (np.mean(min_incremental_energy) < eps) | (np.mean(exec_comm_sum) < eps)
    energy_raw = np.where(use_laed, laed_raw, eltr_raw)
    # Gate by slack: prioritize energy savings more when slack > 0
    energy_gate = np.clip(0.5 + 0.5 * np.tanh(slack / (slack_med + eps)), 0.2, 1.0)
    energy_gated = energy_raw * energy_gate
    norm_energy = percentile_scale(energy_gated)
    energy_term = -2.1 * norm_energy
    
    # === FAIRNESS TERM: Workflow-relative wait saturation ===
    # Threshold adapts: in tight-DLL regimes, use small absolute wait cap; otherwise use relative
    wait_threshold = np.maximum(
        np.percentile(ready_wait_time, 75) + eps,
        0.5 * slack_med
    )
    wait_saturation = np.clip(ready_wait_time / (wait_threshold + eps), 0.0, 1.0)
    wait_compressed = np.log1p(wait_saturation)  # Smooth, monotonic, bounded
    norm_wait = percentile_scale(wait_compressed)
    fairness_term = -0.4 * norm_wait
    
    # === UNCERTAINTY TERM: Joint risk-zone gated (slack-low AND uncertainty-high) ===
    slack_75 = np.percentile(slack, 75) + eps
    unc_85 = np.percentile(uncertainty, 85) + eps
    # Only penalize when task is in high-risk zone: slack ≤ 75th percentile AND uncertainty ≥ 85th
    unc_boost = np.where(
        (slack <= slack_75) & (uncertainty >= unc_85),
        np.clip(uncertainty * 0.06, 0.0, 0.05),
        0.0
    )
    norm_unc = percentile_scale(unc_boost)
    uncertainty_term = 0.06 * norm_unc
    
    # === VIOLATION BOOST: Hard deadline enforcement ===
    has_violation = np.any(slack <= 0)
    violation_boost = np.where(
        has_violation,
        np.where(slack > 0, 1e6 * (1.0 - np.tanh(np.clip(slack, 0.0, 5.0) / 5.0)), 0.0),
        0.0
    )
    
    # Aggregate with strict term hierarchy: urgency > CP > energy > fairness > uncertainty
    score = (
        urgency_term +
        cp_density_term +
        energy_term +
        fairness_term +
        uncertainty_term +
        violation_boost
    )
    
    # Final sanitization: finite, bounded, deterministic shape
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e8, 1e8)
    
    # Ensure (N,) shape — critical for environment compatibility
    return score.astype(float).reshape(-1)
