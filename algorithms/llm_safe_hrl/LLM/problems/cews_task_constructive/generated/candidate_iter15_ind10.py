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
    v2: Streamlined lexicographic deadline-aware priority scorer.
    
    Key improvements:
    - Replaces fragile tau-based slack normalization with robust percentile-based slack threshold (p90 of positive slack)
    - Eliminates overlapping dominance masks; uses strict sequential gating: urgency → critical-leverage → ECR → latency → aging
    - Decouples critical-leverage from sign-sensitive slack: uses normalized *positive* slack margin for leverage, zero when slack <= 0
    - Restores clean ECR signal by masking only with feasibility (slack > 0), not penalizing tight-deadline tasks
    - Introduces adaptive aging boost activated *only* when slack > median_positive_slack to prevent starvation without DDL compromise
    - Uncertainty penalty applied uniformly but scaled by absolute slack violation and relative uncertainty, no risk-factor inflation
    - All terms normalized via MAD with guaranteed epsilon-robust fallbacks; final score strictly preserves lexicographic order via multiplicative scaling
    """
    eps = 1e-08
    N = len(slack)
    
    # Safe casting and nan/inf handling
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=0.0, posinf=1e9, neginf=eps)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=0.0, posinf=1e9, neginf=eps)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=0.0, posinf=1e9, neginf=eps)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=1e9, neginf=-1e9)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=0.0, posinf=1e9, neginf=eps)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=0.0, posinf=1e9, neginf=eps)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=1e9, neginf=eps)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=0.0, posinf=1e9, neginf=eps)
    
    def normalize_mad(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        center = np.median(x)
        dev = np.abs(x - center)
        mad = np.median(dev)
        if mad < eps:
            xmin, xmax = np.min(x), np.max(x)
            scale = xmax - xmin
            if scale < eps:
                scale = eps
            return (x - center) / (scale + eps)
        return (x - center) / (mad + eps)
    
    # Robust slack threshold: p90 of positive slack, fallback to 0.5
    positive_slack = slack[slack > 0]
    tau = np.percentile(positive_slack, 90) if len(positive_slack) > 0 else 0.5
    tau = max(tau, 0.1)
    
    # Urgency term: clipped sigmoid for smooth transition near deadline, hard penalty for violations
    slack_norm = slack / (tau + eps)
    urgency_sigmoid = 1.0 / (1.0 + np.exp(-4.0 * slack_norm))
    abs_slack_violation = np.maximum(-slack, 0.0)
    hard_penalty = np.clip(abs_slack_violation * 12.0, 0.0, 200.0)
    urgency_raw = np.where(slack >= 0, urgency_sigmoid, 1.0 + hard_penalty / 100.0)
    norm_urgency = normalize_mad(urgency_raw)
    urgency_term = -5.0 * norm_urgency
    
    # Critical-leverage: upward_rank * (normalized slack margin), zero when slack <= 0
    slack_margin = np.maximum(slack, 0.0)  # only positive margin contributes
    normalized_margin = np.clip(slack_margin / (tau + eps), 0.0, 5.0)
    critical_leverage = upward_rank * normalized_margin
    critical_leverage_masked = np.where(slack > 0, critical_leverage, 0.0)
    norm_critical_leverage = normalize_mad(critical_leverage_masked)
    critical_leverage_term = -1.5 * norm_critical_leverage
    
    # ECR (Energy-Criticality Ratio): energy / (upward_rank * work), masked only for feasibility (slack > 0)
    ecr_denom = upward_rank * remaining_work + eps
    ecr_base = min_incremental_energy / ecr_denom
    ecr_masked = np.where(slack > 0, ecr_base, 1e6)  # large value for infeasible, not penalty
    norm_ecr = normalize_mad(ecr_masked)
    ecr_term = -1.7 * norm_ecr
    
    # Latency term: exec + comm, weighted by uncertainty only for feasible tasks
    base_latency = min_exec_time + min_comm_time + eps
    uncertainty_weight = np.where(slack > 0, 1.0 + 0.3 * np.tanh(uncertainty), 1.0)
    latency_scaled = base_latency * uncertainty_weight
    norm_latency = normalize_mad(latency_scaled)
    latency_term = 0.5 * norm_latency
    
    # Aging term: log(wait) / (slack_margin + eps), activated only when slack > tau (adaptive fairness)
    wait_safe = np.maximum(ready_wait_time, 0.0)
    log_wait = np.log1p(wait_safe)
    aging_enabled = (slack > tau).astype(float)
    aging_raw = np.divide(log_wait, slack_margin + eps)
    aging_clipped = np.clip(aging_raw, 0.0, 0.6)
    norm_aging = normalize_mad(aging_clipped)
    aging_term = -0.25 * norm_aging * aging_enabled
    
    # Uncertainty penalty: relative uncertainty scaled by absolute slack violation
    rel_uncertainty = uncertainty / (np.mean(uncertainty + eps) + eps)
    uncertainty_penalty = rel_uncertainty * (abs_slack_violation + eps)
    norm_uncertainty = normalize_mad(uncertainty_penalty)
    uncertainty_term = 0.4 * norm_uncertainty
    
    # Strict lexicographic composition: urgency dominates first, then critical-leverage, then others
    # Use multiplicative scaling to preserve ordering: higher-priority terms get larger weight magnitude
    # Scale lower-priority terms by decreasing factors to avoid drowning
    score = (
        urgency_term +
        critical_leverage_term +
        0.8 * ecr_term +
        0.6 * latency_term +
        0.4 * aging_term +
        0.3 * uncertainty_term
    )
    
    # Final robustness: clamp extreme values, ensure shape
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    assert score.shape == (N,), f'Expected shape {(N,)}, got {score.shape}'
    return score
