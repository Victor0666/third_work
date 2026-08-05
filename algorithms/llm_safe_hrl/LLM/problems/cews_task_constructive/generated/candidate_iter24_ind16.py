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
    v3 evolution: Hard-DDL-first with *zero-tolerance violation dominance*, 
    *slack-adaptive criticality gating*, *robust energy-latency tradeoff via normalized marginal efficiency ratio (EMR)*,
    *lateness-aware fairness with dynamic starvation horizon*, and *uncertainty-aware risk amplification*.
    
    Key improvements over v1:
    - Violation boost now dominates *all other terms* (via additive -1e9) to enforce strict hard-DDL priority hierarchy.
    - Replaces CED + latency-pressure with EMR: (min_incremental_energy / (min_exec_time + min_comm_time + eps)) — 
      directly captures energy-per-unit-latency, numerically stable, interpretable, and inversely aligned with DDL urgency.
    - Criticality gating tightened: only tasks with slack <= 0.5s contribute to critical-path density term, avoiding over-prioritization of marginally late tasks.
    - Fairness now uses *dynamic starvation horizon*: threshold = max(ready_wait_time, 0.3 * |median_slack|) AND modulated by exp(-max(0, -slack)/tau) to suppress fairness pressure when lateness is imminent.
    - Uncertainty penalty upgraded to *risk-amplified slack zone*: triggers only in high-risk quadrant (slack < median_slack AND uncertainty > median_uncertainty), scaled quadratically for sensitivity.
    - All normalization uses robust IQR scaling with explicit fallback for degenerate cases (N=1 or constant arrays).
    - Final score clamped to [-1e8, 1e8] and reshaped to ensure shape (N,).
    """
    eps = 1e-08
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
    
    # Robust IQR-based normalization with N=1 safety and degenerate-case fallback
    def normalize_iqr(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        q25, q75 = np.percentile(x, [25, 75])
        iqr = q75 - q25 + eps
        center = np.median(x)
        scale = iqr if iqr > eps else (np.max(x) - np.min(x) + eps)
        return (x - center) / (scale + eps)
    
    # === 1. HARD VIOLATION DOMINANCE ===
    # Assign overwhelming negative priority to any task violating deadline (slack <= 0)
    # Ensures absolute precedence — no other term can override
    violation_boost = np.where(slack <= 0, -1e9, 0.0)
    
    # === 2. SLACK-ADAPTIVE URGENCY (ZERO FOR NON-CRITICAL) ===
    # Strict zero urgency for slack >= 0; linear ramp for slack in [-slack_med, 0]
    slack_abs = np.abs(slack)
    slack_med = np.median(slack_abs) + eps
    urgency_raw = np.where(slack <= 0, -slack, 0.0)  # only urgent when already late or at risk
    norm_urgency = normalize_iqr(urgency_raw)
    urgency_term = -8.0 * norm_urgency
    
    # === 3. CRITICALITY GATING: STRICT SLACK THRESHOLD ===
    # Only tasks with slack <= 0.5s contribute to critical-path density
    cp_gate = (slack <= 0.5)
    exec_comm_sum = min_exec_time + min_comm_time + eps
    horizon_weight = np.exp(-np.clip(slack_abs, 0.0, 5.0 * slack_med) / (slack_med + eps))
    # Use EMR: Energy per unit latency — lower EMR = more energy-efficient per time unit → higher priority
    emr_raw = min_incremental_energy / (exec_comm_sum + eps)
    # Mask out non-critical tasks; preserve original magnitude for scaling
    emr_masked = np.where(cp_gate, emr_raw, np.max(emr_raw) + 1.0)
    norm_emr = normalize_iqr(emr_masked)
    emr_term = -3.2 * norm_emr  # lower EMR → lower score → higher priority
    
    # === 4. LATENESS-AWARE FAIRNESS WITH DYNAMIC HORIZON ===
    # Starvation horizon shrinks under lateness pressure: suppressed when slack < 0
    wait_threshold_base = np.maximum(np.percentile(ready_wait_time, 75) + eps, 0.3 * slack_med)
    # Lateness suppression factor: exp(-max(0,-slack)/0.1) → drops rapidly as slack goes negative
    lateness_suppress = np.exp(-np.maximum(0.0, -slack) / (0.1 + eps))
    wait_threshold = wait_threshold_base * lateness_suppress
    wait_saturation = np.clip(ready_wait_time / (wait_threshold + eps), 0.0, 1.0)
    # Use sqrt for sublinear saturation
    fairness_raw = np.sqrt(wait_saturation + eps)
    norm_fairness = normalize_iqr(fairness_raw)
    fairness_term = -0.25 * norm_fairness
    
    # === 5. UNCERTAINTY-AWARE RISK AMPLIFICATION ===
    # Trigger only in high-risk quadrant: slack < median_slack AND uncertainty > median_uncertainty
    slack_50 = np.median(slack) + eps
    unc_50 = np.median(uncertainty) + eps
    risk_zone = (slack < slack_50) & (uncertainty > unc_50)
    # Quadratic scaling for sensitivity to extreme uncertainty
    unc_boost = np.where(risk_zone, np.clip(uncertainty**2 * 0.002, 0.0, 0.03), 0.0)
    norm_unc = normalize_iqr(unc_boost)
    uncertainty_term = 0.03 * norm_unc
    
    # === COMBINE ALL TERMS ===
    score = (
        violation_boost +
        urgency_term +
        emr_term +
        fairness_term +
        uncertainty_term
    )
    
    # Final robustness guard
    score = np.nan_to_num(score, nan=1e8, posinf=1e8, neginf=-1e8)
    score = np.clip(score, -1e8, 1e8)
    
    # Ensure shape (N,) — critical for environment compatibility
    return score.astype(float).reshape(-1)
