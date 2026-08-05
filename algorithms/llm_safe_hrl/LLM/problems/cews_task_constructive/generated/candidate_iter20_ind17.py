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
    # Sanitize inputs: ensure finite, non-NaN, bounded values
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
    
    # Robust slack: account for uncertainty margin (Parent 2 strength)
    robust_slack = slack - 2.0 * uncertainty
    robust_slack = np.clip(robust_slack, -1e12, 1e12)
    
    # Piecewise deadline urgency: exact penalty for negative robust_slack,
    # linear decay for small positive slack, zero beyond tau_urgency (Parent 1 evolution)
    tau_urgency = 1.0
    urgency_raw = np.where(
        robust_slack <= 0.0,
        -robust_slack * 6.0,  # Stronger penalty for violation
        np.where(
            robust_slack <= tau_urgency,
            6.0 * (1.0 - robust_slack / (tau_urgency + eps)),
            0.0
        )
    )
    
    # Energy-latency ratio with physics-informed gating (hybrid: Parent 2 gate + Parent 1 ELR structure)
    total_latency = min_exec_time + min_comm_time + eps
    energy_gate = np.exp(-np.maximum(0.0, -robust_slack) / (0.1 + eps))  # Parent 2 gating
    elr_base = np.clip(min_incremental_energy / (total_latency + 3.0), 0.0, 1e8)  # Parent 1 tau_energy
    elr_masked = elr_base * energy_gate
    
    # Critical path density: weighted by normalized slack margin to avoid over-prioritizing deep CP when safe
    cp_density = (upward_rank + eps) * (remaining_work + eps) / (total_latency + eps)
    slack_margin_norm = np.clip(np.maximum(0.0, robust_slack) / (tau_urgency + eps), 0.0, 1.0)
    cp_density_scaled = cp_density * slack_margin_norm
    
    # Fairness term: relative wait time gated by slack surplus and latency-normalized (Parent 1 fairness + Parent 2 conditionality)
    rel_wait = np.clip(ready_wait_time / (1.0 + np.mean(total_latency) if len(total_latency) > 0 else 1.0), 0.0, 10.0)
    fairness_boost = np.where(
        (robust_slack > 0.05) & (rel_wait > 0.3),
        np.clip(0.15 * (rel_wait - 0.3), 0.0, 0.15),
        0.0
    )
    
    # Uncertainty risk boost: additive, strictly gated (Parent 1 cap + Parent 2 condition)
    unc_boost = np.where(
        (robust_slack < -0.1) & (uncertainty > 0.15),
        np.clip(uncertainty * np.abs(robust_slack) ** 0.75, 0.0, 0.55),
        0.0
    )
    
    # MAD-based normalization (Parent 2 strength): robust, outlier-resistant, works for N=1
    def safe_mad_normalize(x):
        if x.size == 0:
            return np.zeros_like(x)
        if x.size == 1:
            return np.zeros_like(x)
        median_x = np.median(x)
        mad = np.median(np.abs(x - median_x)) + eps
        if mad < eps:
            return np.zeros_like(x)
        normed = (x - median_x) / mad
        return np.clip(normed, -3.0, 3.0)  # Wider clipping than Parent 2 for stability
    
    norm_urgency = safe_mad_normalize(urgency_raw)
    norm_elr = safe_mad_normalize(elr_masked)
    norm_cp = safe_mad_normalize(cp_density_scaled)
    norm_fair = safe_mad_normalize(fairness_boost)
    norm_unc = safe_mad_normalize(unc_boost)
    
    # Weighted combination: emphasize deadline adherence first, then energy, then fairness and risk
    # Coefficients tuned to reflect objective: DDL-hardened → energy-optimal → starvation-avoidant
    score = (
        +7.2 * norm_urgency      # Highest weight: strict deadline enforcement
        - 4.0 * norm_elr         # Strong energy optimization under safety
        - 1.5 * norm_cp          # Moderate critical-path guidance
        - 0.4 * norm_fair        # Light fairness encouragement (avoids starvation only when safe)
        + 0.25 * norm_unc        # Small penalty for high-risk tasks near violation
    )
    
    # Final sanitization: clamp and replace NaN/inf
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    
    # Ensure shape (N,) — reshape handles scalar edge case via broadcasting-safe conversion
    return score.reshape(-1)
