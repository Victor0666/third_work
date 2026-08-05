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
    v2 evolution: Hard-DDL-first with adaptive urgency, critical-path density gating,
    dual-energy robustness, and starvation-aware fairness.
    
    Key synthesis:
    - Uses Parent 2's adaptive slack-gradient urgency (smooth decay beyond median) but adds Parent 1's strict zero-urgency floor for slack >= 0.
    - Combines Parent 2's dynamic criticality horizon with Parent 1's CED base (upward_rank * remaining_work / energy) and tight slack gating (slack >= -0.2s).
    - Replaces Parent 2's ELTR/LAED fallback with Parent 1's numerically stable CED + latency-pressure term scaled by upward_rank and uncertainty.
    - Enhances fairness using Parent 2's workflow-relative saturation threshold but anchored to max(ready_wait_time, 0.5 * |slack|_median) and modulated by lateness risk.
    - Adds explicit violation boost only for slack <= 0, not soft penalty — enforcing hard DDL priority hierarchy.
    - All normalization uses IQR-based percentile scaling with N=1 safety and explicit finite bounds.
    """
    eps = 1e-08
    N = len(slack)
    
    # Clean and sanitize inputs
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
    
    # Robust adaptive normalization (IQR with fallback for degenerate cases)
    def normalize_iqr(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        q25, q75 = np.percentile(x, [25, 75])
        iqr = q75 - q25 + eps
        center = np.median(x)
        scale = iqr if iqr > eps else (np.max(x) - np.min(x) + eps)
        return (x - center) / (scale + eps)
    
    # === URGENCY TERM: Hard-DDL first with adaptive gradient ===
    # Strict zero when slack >= 0; linear penalty when slack < 0; smooth decay beyond local slack median for positive slack
    slack_abs = np.abs(slack)
    slack_med = np.median(slack_abs) + eps
    urgency_raw = np.where(
        slack <= 0,
        -slack,  # linear penalty for violations
        np.where(
            slack <= slack_med,
            0.0,  # zero urgency for safe slack (Parent 1 principle)
            np.clip((slack - slack_med) / (slack_med + eps), 0.0, 1.0) * 0.1  # mild de-prioritization far from deadline
        )
    )
    norm_urgency = normalize_iqr(urgency_raw)
    urgency_term = -10.0 * norm_urgency  # dominant term: smaller score = higher priority
    
    # === CRITICAL-PATH DENSITY TERM (CED) with horizon gating ===
    # Parent 1 CED base: upward_rank * remaining_work / (energy + eps)
    # Parent 2 horizon: exp(-|slack|/τ), τ = slack_med
    horizon_weight = np.exp(-np.clip(slack_abs, 0.0, 10.0 * slack_med) / (slack_med + eps))
    ced_base = upward_rank * remaining_work * horizon_weight / (min_incremental_energy + eps)
    # Tight gating: only active when slack >= -0.2s (Parent 1’s tau_strict)
    tau_strict = 0.2
    ced_masked = np.where(slack >= -tau_strict, ced_base, 0.0)
    norm_ced = normalize_iqr(ced_masked)
    ced_term = -2.5 * norm_ced  # promote energy-critical paths early, but subordinate to urgency
    
    # === LATENCY-PRESSURE TERM: penalizes high-latency tasks under tight slack ===
    # (exec + comm) / (|slack| + eps) * upward_rank * (1 + uncertainty)
    exec_comm_sum = min_exec_time + min_comm_time + eps
    latency_pressure_base = exec_comm_sum / (np.abs(slack) + eps)
    latency_pressure_scaled = latency_pressure_base * upward_rank * (1.0 + np.clip(uncertainty, 0.0, 5.0))
    norm_latency_pressure = normalize_iqr(latency_pressure_scaled)
    latency_pressure_term = 0.8 * norm_latency_pressure  # secondary penalty for latency-sensitive tasks
    
    # === FAIRNESS TERM: starvation relief gated by lateness risk ===
    # Saturation threshold: max(75th percentile wait, 0.5 * slack_med) → Parent 2 anchor + Parent 1 risk modulation
    wait_p75 = np.percentile(ready_wait_time, 75) + eps
    wait_threshold = np.maximum(wait_p75, 0.5 * slack_med)
    wait_saturation = np.clip(ready_wait_time / (wait_threshold + eps), 0.0, 1.0)
    # Amplify relief when slack is negative (Parent 1 insight): sqrt(wait) * exp(-uncertainty) / (1 + max(0,-slack) + eps)
    wait_safe = np.maximum(ready_wait_time, 0.0)
    sqrt_wait = np.sqrt(wait_safe + eps)
    risk_decay = np.exp(-np.clip(uncertainty, 0.0, 10.0))
    fairness_numerator = sqrt_wait * risk_decay
    fairness_denominator = 1.0 + np.maximum(0.0, -slack) + eps
    fairness_raw = fairness_numerator / fairness_denominator
    norm_fairness = normalize_iqr(fairness_raw)
    fairness_term = -0.3 * norm_fairness  # gentle boost for long-waiting tasks, stronger when lateness risk exists
    
    # === UNCERTAINTY PENALTY: joint risk zone detection ===
    # Only activates when slack is in bottom 75% AND uncertainty in top 85%
    slack_75 = np.percentile(slack, 75) + eps
    unc_85 = np.percentile(uncertainty, 85) + eps
    unc_boost = np.where(
        (slack <= slack_75) & (uncertainty >= unc_85),
        np.clip(uncertainty * 0.05, 0.0, 0.04),
        0.0
    )
    norm_unc = normalize_iqr(unc_boost)
    uncertainty_term = 0.04 * norm_unc
    
    # === HARD VIOLATION BOOST: absolute priority for violating tasks ===
    # Assign massive negative score to any task with slack <= 0 to guarantee selection before others
    has_violation = np.any(slack <= 0)
    violation_boost = np.where(
        slack <= 0,
        -1e6,  # dominates all other terms
        0.0
    )
    
    # Combine terms with strict hierarchy: violation > urgency > CED > latency > fairness > uncertainty
    score = (
        violation_boost +
        urgency_term +
        ced_term +
        latency_pressure_term +
        fairness_term +
        uncertainty_term
    )
    
    # Final sanitization: ensure finite, deterministic, shape-(N,)
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e8, 1e8)
    return score.astype(float).reshape(-1)
