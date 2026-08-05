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
    # Core principle: hard deadline dominance → energy optimization only when safe
    # Robust slack: conservative deadline margin accounting for uncertainty
    eps = 1e-08
    def sanitize(x):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(x, nan=eps, posinf=1e12, neginf=eps)
    
    min_exec_time = sanitize(min_exec_time)
    min_comm_time = sanitize(min_comm_time)
    min_incremental_energy = sanitize(min_incremental_energy)
    slack = sanitize(slack)
    upward_rank = sanitize(upward_rank)
    remaining_work = sanitize(remaining_work)
    ready_wait_time = sanitize(ready_wait_time)
    uncertainty = sanitize(uncertainty)
    
    # Robust slack: pessimistic margin = slack - 2*uncertainty (worst-case delay buffer)
    robust_slack = slack - 2.0 * uncertainty
    robust_slack = np.clip(robust_slack, -1e12, 1e12)
    
    # Deadline penalty: smooth, strictly monotonic, zero when robust_slack > 0.1
    # Quadratic ramp for severe lateness + linear ramp near soft threshold
    deadline_penalty = (
        np.where(robust_slack > 0.1, 0.0,
                 np.where(robust_slack > 0.0, 0.5 * (0.1 - robust_slack), 0.0)) 
        + np.maximum(0.0, -robust_slack) ** 2
    )
    
    # Hard gating for energy optimization: ON only if robust_slack > 0 → strict DDL-first
    slack_gate = np.where(robust_slack > 0.0, 1.0, 0.0)
    
    # SEER (Specific Energy Efficiency Ratio): energy per unit latency
    # Gated to zero when deadline at risk → no energy optimization under violation
    seer_base = (min_incremental_energy + eps) / (min_exec_time + min_comm_time + eps)
    seer_gated = seer_base * slack_gate
    
    # Uncertainty-weighted critical path density: reward high-importance tasks,
    # but suppress low-confidence ones more aggressively than linear scaling
    cp_density = (upward_rank + eps) * (remaining_work + eps) / (1.0 + uncertainty**2 + eps)
    
    # Fairness: normalized wait ratio, activated only when safe AND meaningful wait
    total_latency = min_exec_time + min_comm_time + eps
    wait_ratio = ready_wait_time / total_latency
    fairness_boost = np.where(
        (robust_slack > 0.05) & (wait_ratio > 0.3),
        np.clip(0.2 * np.tanh(2.0 * wait_ratio), 0.0, 0.2),
        0.0
    )
    
    # Risk penalty: only for late + uncertain tasks — targets high-risk violations
    uncertainty_risk = np.where(
        (robust_slack < -0.05) & (uncertainty > 0.1),
        np.clip(uncertainty * np.abs(robust_slack), 0.0, 0.7),
        0.0
    )
    
    # MAD normalization with degenerate handling and tight bounds
    def safe_mad_normalize(x):
        if x.size == 0:
            return np.zeros_like(x)
        if x.size == 1:
            return np.zeros_like(x)
        median_x = np.median(x)
        mad = np.median(np.abs(x - median_x)) + eps
        normed = (x - median_x) / mad
        return np.clip(normed, -2.0, 2.0)
    
    # Normalize all components for balanced weighting
    deadline_norm = safe_mad_normalize(deadline_penalty)
    seer_norm = safe_mad_normalize(seer_gated)
    cp_norm = safe_mad_normalize(cp_density)
    fairness_norm = safe_mad_normalize(fairness_boost)
    risk_norm = safe_mad_normalize(uncertainty_risk)
    
    # Lexicographic priority: deadline dominates, then energy efficiency, then criticality,
    # then fairness, then risk penalty — all scaled to enforce ordering hierarchy
    # Weights tuned to ensure deadline_term >> seer_term >> cp_term >> fairness_term > risk_term
    score = (
        +8.0 * deadline_norm              # strongest pull toward meeting deadlines
        - 4.0 * seer_norm                # maximize energy efficiency only when safe
        - 2.0 * cp_norm                  # prioritize critical-path work
        + 0.1 * fairness_norm            # mild aging to prevent starvation
        + 0.3 * risk_norm                # penalize risky late tasks
    )
    
    # Final sanitization: finite, deterministic, shape (N,)
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    return score.reshape(-1)
