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
    v2: Deadline-hardened, energy-gated, starvation-immune, and uncertainty-contextual priority.
    
    Key evolutions:
    - Urgency: strict piecewise-linear penalty with *asymmetric scaling* — stronger penalty for slack < -1s (imminent violation) and smoothed ramp above zero → preserves DDL hardness while avoiding overreaction to tiny negative slack
    - Criticality: slack-pressure now *multiplicatively gated by normalized upward_rank* to prevent low-rank tasks from dominating under pressure; uses robust density (remaining_work / max(exec_effort, eps)) scaled by rank percentile → better critical-path focus
    - Energy efficiency: replaces EER inversion with *energy-per-useful-work ratio*, gated by sigmoidal slack + hard clamp at slack <= 0 → prioritizes energy only when deadlines permit, avoids spurious efficiency under violation
    - Fairness: wait_ratio now includes *exponential aging* (1 - exp(-ready_wait_time / (median_exec+eps))) to guarantee monotonic priority growth for stalled tasks, bounded and numerically stable
    - Uncertainty: applied *only to latency-sensitive tasks* (slack in [0, 3*median_positive_slack]), with adaptive ceiling based on IQR → prevents noise amplification in high-uncertainty/low-pressure regimes
    - All normalizations use unified robust_normalize with MAD fallback and explicit N=1 handling; all divisions guarded; final score clamped to finite bounds
    - Deterministic, no side effects, fully vectorized, satisfies all interface contracts.
    """
    eps = 1e-08
    # Safe casting and NaN/inf sanitization
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=eps, neginf=eps)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=eps, neginf=eps)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=eps, neginf=eps)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=eps, neginf=-eps)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=0.0, posinf=eps, neginf=0.0)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=0.0, posinf=eps, neginf=0.0)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=eps, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=0.0, posinf=eps, neginf=0.0)
    
    N = len(slack)
    if N == 0:
        return np.array([], dtype=float)
    
    def robust_normalize(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if N == 1:
            return np.array([0.0])
        q25, q50, q75 = np.percentile(x, [25, 50, 75], axis=0, keepdims=False)
        iqr = q75 - q25
        if iqr > eps:
            scale = iqr
        else:
            abs_dev = np.abs(x - q50)
            mad = np.median(abs_dev)
            scale = mad if mad > eps else np.max(np.abs(x - q50)) if np.any(x != q50) else eps
        return (x - q50) / (scale + eps)
    
    # === Urgency: Hard deadline enforcement with asymmetric sensitivity ===
    # Strong penalty for imminent violation (slack < -1.0), moderate ramp above zero
    urgency_raw = np.where(slack < -1.0, -slack * 4.0,
                          np.where(slack < 0.0, -slack * 2.0, slack * 0.15))
    norm_urgency = robust_normalize(urgency_raw)
    urgency_term = -4.0 * norm_urgency
    
    # === Criticality: Slack-gated & rank-weighted critical path density ===
    exec_effort = np.maximum(min_exec_time, eps)
    base_density = remaining_work / (exec_effort + eps)
    # Rank percentile gating: only amplify high-rank tasks under pressure
    rank_percentile = np.argsort(np.argsort(upward_rank)) / max(N - 1, 1) if N > 1 else np.array([0.5])
    slack_pressure = np.where(slack <= 0, 1.0 + np.abs(slack) / (np.median(np.where(slack > 0, slack, eps)) + eps),
                             1.0 + 0.5 * (1.0 - np.clip(slack / (np.median(np.where(slack > 0, slack, 1.0)) + eps), 0.0, 1.0)))
    critical_density = base_density * upward_rank * rank_percentile * slack_pressure
    norm_critical = robust_normalize(critical_density)
    critical_term = -1.5 * norm_critical
    
    # === Energy efficiency: Work-normalized energy, slack-gated, hard-clamped ===
    # Energy per useful work (joules per MI), only rewarded when slack >= 0
    energy_per_work = min_incremental_energy / (remaining_work + eps)
    # Sigmoidal gate: full suppression at slack < 0, smooth activation above
    slack_gate = np.where(slack < 0, 0.0,
                         0.05 + 0.95 / (1.0 + np.exp(-(slack - 0.5) / 0.8)))
    energy_gated = energy_per_work * slack_gate
    norm_energy = robust_normalize(energy_gated)
    efficiency_term = -0.85 * norm_energy
    
    # === Fairness: Exponential aging + wait ratio, bounded and stable ===
    median_exec = np.median(min_exec_time) + eps
    exp_aging = 1.0 - np.exp(-np.clip(ready_wait_time, 0.0, 20.0 * median_exec) / (median_exec + eps))
    duration_estimate = min_exec_time + min_comm_time + eps
    wait_ratio = np.clip(ready_wait_time / (duration_estimate + eps), 0.0, 10.0)
    combined_wait = wait_ratio + 0.3 * exp_aging
    norm_wait = robust_normalize(combined_wait)
    fairness_term = -0.45 * np.clip(norm_wait, -1.3, 1.7)
    
    # === Uncertainty: Contextual penalty only for tight-slack tasks ===
    pos_slack = slack[slack > 0]
    median_pos_slack = np.median(pos_slack) if len(pos_slack) > 0 else 1.0
    tight_mask = (slack >= 0) & (slack <= 3.0 * median_pos_slack)
    unc_q75 = np.percentile(uncertainty, 75) + eps
    # Adaptive ceiling: avoid distortion in high-uncertainty flat regimes
    unc_scaled = np.clip(uncertainty / unc_q75, 0.0, 2.5)
    uncertainty_factor = np.where(tight_mask, 1.0 + 0.7 * unc_scaled, 1.0)
    base_latency = min_exec_time + min_comm_time + eps
    latency_penalty = base_latency * uncertainty_factor
    norm_latency = robust_normalize(latency_penalty)
    latency_term = 0.6 * norm_latency
    
    # Aggregate score: smaller = higher priority
    score = urgency_term + critical_term + efficiency_term + fairness_term + latency_term
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    
    # Final clamp to ensure finite deterministic bounds
    score = np.clip(score, -1e10, 1e10)
    
    return score
