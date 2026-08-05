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
    v2: Lexicographic urgency-first prioritization with smooth deadline-aware efficiency,
    robust fairness gating, and numerically stable normalization.
    
    Key improvements:
    - Urgency: hybrid linear + exponential decay (Parent 2) with hard negative-slack penalty
    - Efficiency: SCEE (slack-gated critical work / latency) replaces ECR for better gradient behavior
    - Latency: uncertainty-weighted only when slack > 0, using tanh-bounded scaling (robust vs. raw mult)
    - Fairness: adaptive wait-boost gated by dynamic slack threshold (median + margin), log-scaled to prevent explosion
    - Normalization: MAD-based with fallback to range scaling; handles N=1 safely and avoids inf/nan
    - Prioritization: strict dominance via urgency rank masking — ensures deadline-critical tasks always win
    """
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    
    def normalize_mad(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        med = np.median(x)
        abs_dev = np.abs(x - med)
        mad = np.median(abs_dev)
        if mad < eps:
            xmin, xmax = np.min(x), np.max(x)
            scale = xmax - xmin
            if scale < eps:
                scale = eps
            return (x - (xmin + xmax) / 2.0) / (scale + eps)
        return (x - med) / (mad + eps)
    
    # === Urgency: dominant priority signal ===
    # Linear penalty for negative slack (immediate violation risk)
    urgency_linear = np.where(slack < 0, -slack, 0.0)
    # Smooth exponential reward for positive slack (encourages early scheduling of tight deadlines)
    urgency_exp = np.where(slack > 0, -np.exp(-slack / 60.0), 0.0)
    urgency_raw = urgency_linear + urgency_exp
    norm_urgency = normalize_mad(urgency_raw)
    urgency_term = -3.2 * norm_urgency
    
    # === Efficiency: Slack-aware Critical-Energy Efficiency (SCEE) ===
    # Numerator: critical path weight (upward_rank * remaining_work)
    # Denominator: base latency (execution + comm), epsilon-protected
    scee_base = (upward_rank * remaining_work) / (min_exec_time + min_comm_time + eps)
    # Sigmoid gate: smoothly activates efficiency signal as slack improves (centered at slack=0)
    gate = 1.0 / (1.0 + np.exp(-slack / 15.0))
    scee_gated = scee_base * gate
    norm_scee = normalize_mad(scee_gated)
    scee_term = -1.6 * norm_scee
    
    # === Latency: uncertainty-penalized only for feasible tasks (slack > 0) ===
    base_latency = min_exec_time + min_comm_time + eps
    # Tanh-bounded uncertainty scaling: prevents explosion, saturates at ~±1 → 0.76–1.24 multiplier
    uncertainty_scale = 1.0 + 0.5 * np.tanh(uncertainty)
    latency_scaled = np.where(slack > 0, base_latency * uncertainty_scale, base_latency)
    norm_latency = normalize_mad(latency_scaled)
    latency_term = 0.65 * norm_latency
    
    # === Fairness: adaptive wait boost only when slack > median_slack + buffer ===
    slack_median = np.median(slack)
    wait_boost_active = (slack > slack_median + 0.1).astype(float)  # small margin avoids jitter
    # Log-linear aging: bounded wait time prevents overflow, monotonic and interpretable
    wait_bounded = np.clip(ready_wait_time, 0.0, 120.0)
    fairness_raw = np.log1p(wait_bounded) * wait_boost_active
    norm_fairness = normalize_mad(fairness_raw)
    fairness_term = -0.12 * norm_fairness
    
    # === Lexicographic dominance: urgency ranks override all others when critical ===
    # Assign integer urgency rank (0=highest urgency); top 10% get absolute priority
    urgency_rank = np.argsort(np.argsort(-urgency_raw))  # descending order → lower rank = higher urgency
    urgency_dominance = (urgency_rank < np.ceil(0.1 * len(urgency_raw))).astype(float)
    non_urgency_scale = 1.0 - urgency_dominance
    
    # Combine: urgency always dominates; others contribute only where urgency is not critical
    score = urgency_term + non_urgency_scale * (scee_term + latency_term + fairness_term)
    
    # Final sanitization: ensure finite, deterministic output
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    
    return score
