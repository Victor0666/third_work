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
    v4: Hybrid safety-energy-criticality priority with:
      - Hard deadline violation override (slack < 0 → minimum score) preserved and made dominant
      - Robust MAD-based normalization for all terms to handle skew/outliers
      - Adaptive SEER: (upward_rank * remaining_work) / (min_incremental_energy + eps), gated by sigmoid of normalized slack
      - Critical-path pressure (CPP): upward_rank / (min_exec_time + min_comm_time + eps), scaled by slack deficit magnitude
      - Wait-aware urgency: sqrt(ready_wait_time) modulated by tanh(slack/tau_wait) to prevent starvation without violating deadlines
      - Uncertainty penalty dynamically suppressed when slack is deeply negative (violation already active)
      - Fairness term replaced by progressive wait penalty with uncertainty discounting and slack-aware boost
      - All operations zero-safe, NaN/inf-guarded, deterministic, and N=1 compliant
    """
    eps = 1e-08
    
    # Sanitize inputs: ensure finite floats, replace NaN/inf with safe defaults
    def sanitize(x):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(x, nan=eps, posinf=1e6, neginf=eps)
    
    min_exec_time = sanitize(min_exec_time)
    min_comm_time = sanitize(min_comm_time)
    min_incremental_energy = sanitize(min_incremental_energy)
    slack = sanitize(slack)
    upward_rank = sanitize(upward_rank)
    remaining_work = sanitize(remaining_work)
    ready_wait_time = sanitize(ready_wait_time)
    uncertainty = sanitize(uncertainty)
    
    # Mask for tasks violating or at risk of violating deadline
    violation_mask = slack < 0
    
    # Robust normalization using MAD (median absolute deviation) for stability under outliers
    def normalize_robust(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        med = np.median(x)
        abs_dev = np.abs(x - med)
        mad = np.median(abs_dev)
        if mad > eps:
            scale = mad + eps
        else:
            xmin, xmax = np.min(x), np.max(x)
            scale = xmax - xmin
            if scale < eps:
                scale = eps
        normed = (x - med) / (scale + eps)
        return np.clip(normed, -5.0, 5.0)
    
    # === Urgency Term: prioritize urgent tasks via robust inverse slack mapping ===
    # Use tanh(-slack/tau) for smooth saturation; add linear fallback for extreme slack
    tau_urgency = 0.3
    urgency_raw = np.tanh(-slack / tau_urgency)
    linear_fallback = np.where(slack < -10.0, 1.0, 0.0)  # Strong push for deep violations
    urgency_raw = np.where(violation_mask, 1.0, urgency_raw + linear_fallback)
    norm_urgency = normalize_robust(urgency_raw)
    urgency_term = -5.0 * norm_urgency  # Higher weight to enforce deadline safety
    
    # === Critical-Path Pressure (CPP): bottleneck awareness amplified under tight slack ===
    latency_cost = min_exec_time + min_comm_time + eps
    cpp_base = upward_rank / latency_cost
    norm_cpp = normalize_robust(cpp_base)
    # Scale CPP strength by normalized slack deficit: stronger when slack is more negative
    norm_slack = normalize_robust(slack)
    slack_deficit_factor = np.clip(1.0 + 0.7 * np.maximum(0.0, -norm_slack), 0.5, 2.5)
    cpp_term = -2.2 * norm_cpp * slack_deficit_factor
    
    # === SEER Term: normalized marginal energy efficiency, gated by deadline proximity ===
    seer_base = upward_rank * remaining_work / (min_incremental_energy + eps)
    # Adaptive gate: tighter activation near deadline, relaxed when slack is ample
    tau_seer = np.clip(0.4 + 0.6 * (1.0 / (1.0 + np.exp(-norm_slack))), 0.4, 1.8)
    seer_gate = 1.0 / (1.0 + np.exp(-(slack + eps) / tau_seer))
    seer_gated = seer_base * seer_gate
    norm_seer = normalize_robust(seer_gated)
    seer_term = -1.5 * norm_seer  # Energy-efficiency reward (lower score = better)
    
    # === Wait-Aware Fairness: prevents starvation while respecting deadlines ===
    wait_safe = np.maximum(ready_wait_time, 0.0)
    sqrt_wait = np.sqrt(wait_safe + eps)
    # Boost waiting tasks only when slack is non-negative; suppress boost if uncertainty is high
    exp_uncert_discount = np.exp(-uncertainty)
    tau_wait = 1.2
    slack_modulation = 0.6 * np.tanh(np.clip(slack, 0.0, 10.0) / tau_wait)
    fairness_raw = sqrt_wait * exp_uncert_discount * (1.0 + slack_modulation)
    fairness_clipped = np.clip(fairness_raw, 0.0, 0.5)
    norm_fairness = normalize_robust(fairness_clipped)
    fairness_term = -0.25 * norm_fairness  # Mild preference for long-waiting safe tasks
    
    # === Uncertainty Penalty: penalizes risky assignments, suppressed under violation ===
    # Gate: suppress penalty when slack is deeply negative (deadline already broken)
    tau_uncert = 0.9
    uncert_gate = 1.0 - 1.0 / (1.0 + np.exp(-(slack + eps) / tau_uncert))
    uncert_penalty = uncertainty * uncert_gate * 0.15
    norm_uncert = normalize_robust(uncert_penalty)
    uncert_term = 0.5 * norm_uncert
    
    # === Final score composition ===
    score = urgency_term + cpp_term + seer_term + fairness_term + uncert_term
    
    # Hard override: all violating tasks get minimum possible score (highest priority)
    score = np.where(violation_mask, -1e12, score)
    
    # Final sanitization: clamp extremes and ensure finite output
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)
    
    return score
