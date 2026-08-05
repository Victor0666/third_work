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
    Lexicographic deadline-aware priority scorer with robust efficiency signaling.
    
    Key improvements:
    - Combines Parent 2's lexicographic urgency gating and ECR with Parent 1's
      aging boost and risk-amplified uncertainty handling
    - Replaces linear slack margin with dynamic tau-based feasibility threshold
    - Uses clipped sigmoid urgency for smooth transition near deadline boundary
    - Introduces critical-leverage term (upward_rank * normalized_slack) gated by slack > 0
    - Applies MAD normalization consistently across all terms, with epsilon-robust fallbacks
    - Enforces strict priority hierarchy: urgency → critical-leverage → ECR → latency → aging → uncertainty
    """
    eps = 1e-08
    N = len(slack)
    
    # Convert and sanitize inputs
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
    
    # Dynamic tau: median positive slack or safe default
    positive_slack = slack[slack > 0]
    tau = np.maximum(np.median(positive_slack) if len(positive_slack) > 0 else 0.5, 0.1)
    
    # Urgency: clipped sigmoid for smooth penalty near deadline, hard penalty for violation
    slack_norm = slack / (tau + eps)
    urgency_sigmoid = 1.0 / (1.0 + np.exp(-3.0 * slack_norm))
    abs_slack_violation = np.maximum(-slack, 0.0)
    hard_penalty = np.clip(abs_slack_violation * 10.0 + (abs_slack_violation / (tau + eps))**1.5 * 4.0, 0.0, 150.0)
    urgency_raw = np.where(slack >= 0, urgency_sigmoid, 1.0 + hard_penalty / 100.0)
    norm_urgency = normalize_mad(urgency_raw)
    urgency_term = -4.0 * norm_urgency
    
    # Critical-leverage: importance × normalized slack, only active for feasible tasks
    normalized_slack = np.clip(slack / (tau + eps), -1.0, 3.0)
    critical_leverage = upward_rank * normalized_slack
    critical_leverage_masked = np.where(slack >= 0, critical_leverage, -1e6)  # low priority for violated
    norm_critical_leverage = normalize_mad(critical_leverage_masked)
    critical_leverage_term = -1.2 * norm_critical_leverage
    
    # Energy-Criticality Ratio (ECR): energy per unit critical work
    ecr_denom = upward_rank * remaining_work + eps
    ecr_base = min_incremental_energy / ecr_denom
    ecr_masked = np.where(slack >= 0, ecr_base, 1e6)  # high penalty for violated
    norm_ecr = normalize_mad(ecr_masked)
    ecr_term = -1.6 * norm_ecr
    
    # Latency: execution + communication, weighted by uncertainty for feasible tasks
    base_latency = min_exec_time + min_comm_time + eps
    uncertainty_weight = np.where(slack >= 0, 1.0 + 0.4 * np.tanh(uncertainty), 1.0)
    latency_scaled = base_latency * uncertainty_weight
    norm_latency = normalize_mad(latency_scaled)
    latency_term = 0.6 * norm_latency
    
    # Aging boost: log-linear wait time activated only when slack > 0.25*tau to prevent starvation
    wait_safe = np.maximum(ready_wait_time, 0.0)
    log_wait = np.log1p(wait_safe)
    slack_margin = np.maximum(slack - 0.25 * tau, 0.0)
    aging_raw = np.divide(log_wait, slack_margin + eps)
    aging_clipped = np.clip(aging_raw, 0.0, 0.5)
    norm_aging = normalize_mad(aging_clipped)
    aging_term = -0.3 * norm_aging
    
    # Uncertainty penalty: amplified by lateness risk and relative uncertainty
    rel_uncertainty = uncertainty / (np.mean(uncertainty + eps) + eps)
    slack_distance = np.maximum(-slack, 0.0) / (tau + eps)
    risk_factor = 1.0 + np.clip(slack_distance, 0.0, 4.0) + np.clip(rel_uncertainty, 0.0, 2.5)
    uncertainty_penalty = rel_uncertainty * risk_factor * (abs_slack_violation + eps)
    norm_uncertainty = normalize_mad(uncertainty_penalty)
    uncertainty_term = 0.5 * norm_uncertainty
    
    # Lexicographic dominance: urgency dominates first, then critical-leverage, then others
    urgency_rank = np.argsort(np.argsort(urgency_term, kind='stable'))
    critical_leverage_rank = np.argsort(np.argsort(critical_leverage_term, kind='stable'))
    
    # Gate lower-priority terms based on top decile of urgency and critical-leverage
    urgency_dominance = (urgency_rank >= N * 0.9).astype(float)
    critical_dominance = (critical_leverage_rank >= N * 0.9).astype(float)
    combined_dominance = np.maximum(urgency_dominance, critical_dominance)
    non_dominant_scale = 1.0 - combined_dominance
    
    # Final score: urgency + critical-leverage always active; others gated
    score = (
        urgency_term +
        critical_leverage_term +
        non_dominant_scale * (ecr_term + latency_term + aging_term + uncertainty_term)
    )
    
    # Final sanitization
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    
    # Ensure shape correctness
    assert score.shape == (N,), f'Expected shape {(N,)}, got {score.shape}'
    return score
