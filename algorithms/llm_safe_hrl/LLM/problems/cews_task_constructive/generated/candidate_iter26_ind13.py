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
    v2: Hybrid deadline-strict, criticality-gated, energy-efficient, and uncertainty-aware priority.
    
    Key innovations:
    - Hard-DDL violation dominance retained *only for truly violated tasks* (slack < -eps), using -1e9 boost to prevent any violation scheduling.
    - Bounded arctan urgency for *near-deadline* tasks (0 <= slack <= 2*tau) to enable smooth tradeoff without inversion.
    - Criticality-energy ratio (CED) enhanced with EMR-inspired marginal efficiency: (upward_rank * remaining_work) / (min_incremental_energy + eps),
      gated by sigmoid over slack to activate only when deadlines are feasible.
    - Uncertainty now multiplicatively amplifies *both* criticality pressure *and* EMR gating, tightening risk coupling.
    - Fairness term uses dynamic starvation horizon scaled by median slack and suppressed under lateness (exp(-max(0,-slack)/tau)).
    - All normalization uses robust IQR with degenerate-case fallback; all outputs bounded and reshaped to (N,).
    """
    eps = 1e-08
    tau = 1.0
    
    # Clean inputs: replace NaN/inf with safe finite values
    def clean(x):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    
    min_exec_time = clean(min_exec_time)
    min_comm_time = clean(min_comm_time)
    min_incremental_energy = clean(min_incremental_energy)
    slack = clean(slack)
    upward_rank = clean(upward_rank)
    remaining_work = clean(remaining_work)
    ready_wait_time = clean(ready_wait_time)
    uncertainty = clean(uncertainty)
    
    N = len(slack)
    
    # --- Violation dominance: absolute priority for already-violated tasks ---
    violation_boost = np.where(slack < -eps, -1000000000.0, 0.0)
    
    # --- Bounded urgency: arctan for near-deadline (0 <= slack <= 2*tau), zero elsewhere ---
    urgency_raw = np.where((slack >= 0.0) & (slack <= 2.0 * tau), 
                          np.arctan(-slack / (tau + eps)), 
                          0.0)
    
    # --- Robust IQR normalization with N=1 and constant-array fallback ---
    def normalize_iqr(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        q25, q75 = np.percentile(x, [25, 75])
        iqr = q75 - q25 + eps
        center = np.median(x)
        scale = iqr if iqr > eps else (np.max(x) - np.min(x) + eps)
        normed = (x - center) / (scale + eps)
        return np.clip(normed, -5.0, 5.0)
    
    norm_urgency = normalize_iqr(urgency_raw)
    urgency_term = -3.2 * norm_urgency
    
    # --- Criticality-Energy-Delay (CED) term: work-importance per marginal joule ---
    # Numerator: criticality-weighted remaining work
    ced_num = upward_rank * remaining_work
    # Denominator: incremental energy (robust)
    ced_denom = min_incremental_energy + eps
    ced_raw = ced_num / ced_denom
    
    # Sigmoid gate: activates CED only when slack is non-negative and uncertainty-moderated
    # Gate widens slightly under high uncertainty (more aggressive energy optimization when safe)
    gate_center = 0.5 * tau * (1.0 + np.clip(uncertainty, 0.0, 2.0))
    gate_width = 0.25 * tau + eps
    gate_sigmoid = 1.0 / (1.0 + np.exp(-(slack - gate_center) / (gate_width + eps)))
    
    ced_masked = ced_raw * gate_sigmoid
    norm_ced = normalize_iqr(ced_masked)
    ced_term = -1.8 * norm_ced
    
    # --- Criticality pressure: execution+comm cost weighted by rank & uncertainty ---
    exec_comm_sum = min_exec_time + min_comm_time + eps
    pressure_base = upward_rank * exec_comm_sum * (1.0 + uncertainty)
    # Linear gate from slack=-tau (full pressure) to slack=+tau (zero pressure)
    pressure_gate = np.clip((tau - slack) / (2.0 * tau + eps), 0.0, 1.0)
    pressure_masked = pressure_base * pressure_gate
    norm_pressure = normalize_iqr(pressure_masked)
    pressure_term = 1.0 * norm_pressure
    
    # --- Fairness: dynamic starvation horizon with lateness suppression ---
    slack_abs = np.abs(slack)
    slack_med = np.median(slack_abs) + eps
    # Horizon: max(75th percentile wait, 0.3 * median |slack|), modulated by exp(-lateness)
    wait_thresh_base = np.maximum(np.percentile(ready_wait_time, 75) + eps, 0.3 * slack_med)
    lateness_suppress = np.exp(-np.maximum(0.0, -slack) / (tau + eps))
    wait_threshold = wait_thresh_base * lateness_suppress
    wait_saturation = np.clip(ready_wait_time / (wait_threshold + eps), 0.0, 1.0)
    fairness_raw = np.sqrt(wait_saturation + eps)
    norm_fairness = normalize_iqr(fairness_raw)
    fairness_term = -0.2 * norm_fairness
    
    # --- Uncertainty risk amplification in violation zone (slack < median_slack AND high uncertainty) ---
    slack_50 = np.median(slack) + eps
    unc_50 = np.median(uncertainty) + eps
    risk_zone = (slack < slack_50) & (uncertainty > unc_50)
    unc_penalty = np.where(risk_zone, np.clip(uncertainty ** 2 * 0.001, 0.0, 0.01), 0.0)
    norm_unc = normalize_iqr(unc_penalty)
    uncertainty_term = 0.02 * norm_unc
    
    # --- Final score: sum all terms, sanitize, clamp, reshape ---
    score = violation_boost + urgency_term + ced_term + pressure_term + fairness_term + uncertainty_term
    score = np.nan_to_num(score, nan=100000000.0, posinf=100000000.0, neginf=-100000000.0)
    score = np.clip(score, -100000000.0, 100000000.0)
    
    # Ensure shape (N,)
    return score.astype(float).reshape(-1)
