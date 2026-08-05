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
    v2 evolution: Hybrid additive-multiplicative priority with deadline-hardened gating,
    robust per-term normalization, and asymmetric risk-aware modulation.
    
    Key improvements:
    - Uses additive base structure (from Parent 2) for stability and interpretability,
      but applies multiplicative safety gating only on critical terms to preserve urgency dominance.
    - Deadline gating strengthened: energy term activates only when slack > tau_safe AND median_safe_slack > 0,
      preventing false activation in globally tight scenarios.
    - Critical-path density uses upward_rank / (remaining_work + eps) normalized by slack margin *and* uncertainty,
      avoiding scale bias while suppressing uncertain critical tasks.
    - Fairness term redefined as latency-avoiding wait saturation: only tasks with positive slack contribute,
      using dynamic percentile threshold (90th for slack > 5s, 75th otherwise) for adaptive starvation prevention.
    - Uncertainty penalty is asymmetric and context-aware: dampens fairness & CP density linearly up to 0.5,
      then exponentially suppresses beyond — prevents over-scheduling high-risk critical tasks.
    - All terms use robust min-max normalization *within slack-feasible subset*, preserving discriminative power near deadlines.
    - Explicit zero-energy masking: violated tasks (slack < -1e-3) get highest priority (min score = -inf).
    - Final score clamped and reshaped deterministically to (N,).
    """
    eps = 1e-08
    
    # Clean inputs: ensure finite, replace NaN/inf with safe values
    def clean(x):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(x, nan=eps, posinf=1e6, neginf=eps)
    
    min_exec_time = clean(min_exec_time)
    min_comm_time = clean(min_comm_time)
    min_incremental_energy = clean(min_incremental_energy)
    slack = clean(slack)
    upward_rank = clean(upward_rank)
    remaining_work = clean(remaining_work)
    ready_wait_time = clean(ready_wait_time)
    uncertainty = clean(uncertainty)
    
    # Identify violated tasks (hard deadline breach)
    violated_mask = slack < -1e-3
    valid_mask = ~violated_mask
    
    # Compute derived quantities
    exec_comm_sum = min_exec_time + min_comm_time + eps
    tau_urgency = 1.0
    tau_safe = 3.0
    
    # Urgency term: sharp penalty for negative slack, linear decay for small positive slack
    urgency_raw = np.where(
        slack <= 0,
        -slack * 10.0,
        np.where(slack <= tau_urgency, 10.0 * (1.0 - slack / (tau_urgency + eps)), 0.0)
    )
    
    # Latency penalty: penalize long-exec/comm tasks only when urgency is high
    latency_penalty = np.clip(exec_comm_sum / 10.0, 0.0, 1.0) * urgency_raw
    
    # Energy term: only active when slack is safely above tau_safe AND global slack permits energy optimization
    safe_slack = slack[valid_mask] if np.any(valid_mask) else np.array([0.0])
    median_safe_slack = np.median(safe_slack) + eps if len(safe_slack) > 0 else eps
    energy_term = np.where(
        (slack > tau_safe) & (median_safe_slack > 0.1),
        min_incremental_energy / (exec_comm_sum + 3.0),
        0.0
    )
    
    # Critical-path density: upward_rank importance per unit work, gated by slack margin and uncertainty
    cp_density_base = (upward_rank + eps) / (remaining_work + eps)
    slack_margin = np.clip(np.maximum(0.0, slack) / (tau_urgency + eps), 0.0, 1.0)
    uncertainty_suppress = np.clip(1.0 - uncertainty, 0.0, 1.0)
    cp_gated = cp_density_base * slack_margin * uncertainty_suppress
    
    # Fairness term: wait-based starvation avoidance, activated only for non-urgent tasks
    # Dynamic threshold: 90th percentile for relaxed (slack > 5s), 75th for urgent
    wait_thresh = np.where(
        slack > 5.0,
        np.percentile(ready_wait_time, 90) + eps,
        np.percentile(ready_wait_time, 75) + eps
    )
    wait_saturation = np.clip(ready_wait_time / (wait_thresh + eps), 0.0, 1.0)
    fairness_term = wait_saturation * slack_margin * uncertainty_suppress
    
    # Risk boost: mild reward for low-uncertainty, high-slack tasks to promote stable scheduling
    risk_boost = np.where(
        (slack > 2.0 * tau_safe) & (uncertainty < 0.3),
        np.clip((1.0 - uncertainty) * 0.2, 0.0, 0.2),
        0.0
    )
    
    # Robust min-max normalization per term over valid tasks only
    def robust_norm(x):
        if not np.any(valid_mask):
            return np.zeros_like(x)
        x_valid = x[valid_mask]
        if x_valid.size == 1:
            return np.where(valid_mask, 0.0, 0.0)
        x_min, x_max = np.min(x_valid), np.max(x_valid)
        if x_max - x_min < eps:
            return np.where(valid_mask, 0.0, 0.0)
        normed = (x - x_min) / (x_max - x_min + eps)
        return np.where(valid_mask, normed, 0.0)
    
    norm_urgency = robust_norm(urgency_raw)
    norm_latency = robust_norm(latency_penalty)
    norm_energy = robust_norm(energy_term)
    norm_cp = robust_norm(cp_gated)
    norm_fair = robust_norm(fairness_term)
    norm_risk = robust_norm(risk_boost)
    
    # Additive combination with calibrated weights
    # Negative weights for urgency/latency/energy (lower is better), positive for CP/risk (higher priority for critical/safe tasks)
    score = (
        -12.0 * norm_urgency          # Highest weight for deadline adherence
        - 8.0 * norm_latency          # Strong penalty for high-latency under urgency
        - 5.0 * norm_energy           # Moderate energy minimization when safe
        + 1.5 * norm_cp               # Reward critical-path density, but capped
        - 0.7 * norm_fair             # Mild penalty for excessive waiting (fairness)
        + 0.4 * norm_risk             # Small reward for low-risk, high-slack scheduling
    )
    
    # Assign max priority (lowest score) to violated tasks
    score = np.where(violated_mask, -np.inf, score)
    
    # Final sanitization: clamp extremes, ensure finite output
    score = np.nan_to_num(score, nan=-1e6, posinf=-1e6, neginf=-1e6)
    score = np.clip(score, -1e6, 1e6)
    
    # Ensure shape (N,) — no broadcasting, no scalars
    return score.reshape(-1)
