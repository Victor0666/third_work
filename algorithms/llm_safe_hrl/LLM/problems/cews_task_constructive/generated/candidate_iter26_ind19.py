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
    # Robust input sanitization: handle NaN, inf, and extreme values
    eps = 1e-08
    N = len(slack)
    
    def clean_array(x):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(x, nan=eps, posinf=1e6, neginf=eps)
    
    min_exec_time = clean_array(min_exec_time)
    min_comm_time = clean_array(min_comm_time)
    min_incremental_energy = clean_array(min_incremental_energy)
    slack = clean_array(slack)
    upward_rank = clean_array(upward_rank)
    remaining_work = clean_array(remaining_work)
    ready_wait_time = clean_array(ready_wait_time)
    uncertainty = clean_array(uncertainty)
    
    # Quantile-based robust normalization (from Parent 2) — stable for small N
    def quantile_scale(x):
        if x.size == 0:
            return np.zeros_like(x)
        if x.size == 1:
            return np.array([0.0])
        q1 = np.quantile(x, 0.25) + eps
        q3 = np.quantile(x, 0.75) + eps
        iqr = q3 - q1
        scaled = (x - (q1 + q3) / 2) / (iqr * 0.5 + eps)
        return np.clip(scaled, -1.5, 1.5)  # Wider clipping than Parent 2 for stability
    
    # --- Urgency term: DDL-hardness first, with smooth escalation ---
    # Negative slack → high urgency; use tanh for bounded, differentiable penalty growth
    slack_abs = np.abs(slack)
    slack_med = np.median(slack_abs) + eps
    urgency_raw = np.where(
        slack <= 0,
        1.0 + 0.5 * np.tanh(-slack / (slack_med + eps)),  # Stronger boost near deadline
        np.clip(1.0 - (slack_abs - slack_med) / (slack_med + eps), 0.0, 1.0)
    )
    norm_urgency = quantile_scale(urgency_raw)
    urgency_term = -18.0 * norm_urgency  # Higher weight than v1 to enforce hard-DDL priority
    
    # --- Critical-path density (CPD): prioritizes high-impact, low-latency tasks ---
    exec_comm_sum = min_exec_time + min_comm_time + eps
    # Horizon shrinkage only when urgent: exponential decay on negative slack
    horizon_weight = np.where(
        slack <= 0,
        np.exp(np.clip(-slack / (slack_med + eps), 0.0, 4.0)),
        1.0
    )
    # CPD = (upward_rank * remaining_work * horizon_weight) / (exec_comm_sum)
    cp_density_raw = (upward_rank + eps) * (remaining_work + eps) * horizon_weight / exec_comm_sum
    # Add communication uncertainty penalty only when slack > 0 (to avoid over-penalizing urgent tasks)
    comm_unc_penalty = np.where(
        slack > 0,
        min_comm_time * np.clip(uncertainty, 0.0, 5.0) * 0.12,
        0.0
    )
    cp_density_penalized = cp_density_raw + comm_unc_penalty / exec_comm_sum
    norm_cp_density = quantile_scale(cp_density_penalized)
    cp_density_term = 3.5 * norm_cp_density  # Slightly higher than v1 for stronger criticality bias
    
    # --- Energy-latency efficiency (ELTR) with fallback and gating ---
    eltr_raw = (min_incremental_energy + eps) / (exec_comm_sum + eps)
    # Fallback to LAED (energy/work/latency) when ELTR variance is negligible or energy is near-zero
    eltr_var = np.var(eltr_raw) if N > 1 else 0.0
    laed_raw = (min_incremental_energy + eps) / ((remaining_work + eps) * (exec_comm_sum + eps))
    use_laed = (eltr_var < 1e-08) | (np.mean(min_incremental_energy) < eps) | (np.mean(exec_comm_sum) < eps)
    energy_raw = np.where(use_laed, laed_raw, eltr_raw)
    # Energy gating: reduce weight when slack is safe; full weight under pressure
    energy_gate = np.clip(
        0.1 + 0.9 * (1.0 - np.tanh(np.clip(slack_abs, 0.0, 10.0 * slack_med) / (slack_med + eps))),
        0.1, 1.0
    )
    energy_gated = energy_raw * energy_gate
    norm_energy = quantile_scale(energy_gated)
    energy_term = -2.8 * norm_energy  # Stronger energy-efficiency incentive than v1
    
    # --- Fairness: slack-relative waiting time, with saturation-aware compression ---
    wait_rel = ready_wait_time / (slack_abs + eps)  # Normalize wait by deadline pressure
    # Use adaptive threshold: 75th percentile or minimum safe value (0.25)
    wait_threshold = np.maximum(np.quantile(wait_rel, 0.75) + eps, 0.25)
    wait_saturation = np.clip(wait_rel / (wait_threshold + eps), 0.0, 1.0)
    # Log-compression preserves discrimination at low waits, saturates high waits
    wait_compressed = np.log1p(wait_saturation * 2.0)  # Gentle scaling
    norm_wait = quantile_scale(wait_compressed)
    fairness_term = -0.5 * norm_wait  # Slightly stronger fairness than v1
    
    # --- Uncertainty gating: only activate under joint risk (slack < median AND unc > median)
    slack_q50 = np.quantile(slack, 0.5) + eps
    unc_q50 = np.quantile(uncertainty, 0.5) + eps
    unc_boost = np.where(
        (slack < slack_q50) & (uncertainty > unc_q50),
        np.clip(uncertainty * 0.08, 0.0, 0.07),
        0.0
    )
    norm_unc = quantile_scale(unc_boost)
    uncertainty_term = 0.07 * norm_unc
    
    # --- Violation penalty: smooth, unbounded tanh gradient (no hard clipping) ---
    # Penalty grows continuously as slack → -∞, but saturates at ~100K
    violation_raw = np.clip(-slack, 0.0, 5.0 * slack_med)
    violation_penalty = np.tanh(violation_raw / (slack_med + eps)) * 120000.0
    violation_term = violation_penalty
    
    # --- Composite score: lower = better (urgency and efficiency dominate) ---
    score = (
        urgency_term +
        cp_density_term +
        energy_term +
        fairness_term +
        uncertainty_term +
        violation_term
    )
    
    # Final robustness: clamp extremes and ensure finite output
    score = np.nan_to_num(score, nan=1e8, posinf=1e8, neginf=-1e8)
    score = np.clip(score, -1e7, 1e7)
    
    # Ensure shape (N,) — no scalar or column vector
    return score.astype(float).reshape(-1)
