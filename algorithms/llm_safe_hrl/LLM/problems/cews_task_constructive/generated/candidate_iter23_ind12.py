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
    eps = 1e-08
    
    # Sanitize inputs: convert to float, replace NaN/inf/neg-inf with safe values
    def sanitize(x):
        x = np.asarray(x, dtype=float)
        x = np.nan_to_num(x, nan=eps, posinf=1e6, neginf=-1e6)
        return x
    
    min_exec_time = sanitize(min_exec_time)
    min_comm_time = sanitize(min_comm_time)
    min_incremental_energy = sanitize(min_incremental_energy)
    slack = sanitize(slack)
    upward_rank = sanitize(upward_rank)
    remaining_work = sanitize(remaining_work)
    ready_wait_time = sanitize(ready_wait_time)
    uncertainty = sanitize(uncertainty)
    
    # Robust normalization using IQR + median (from Parent 2) — stable for N=1 and outliers
    def robust_normalize(x):
        if x.size == 0:
            return np.zeros_like(x)
        if x.size == 1:
            return np.zeros_like(x)
        q25, q75 = np.percentile(x, [25, 75])
        iqr = q75 - q25 + eps
        med = np.median(x)
        normed = (x - med) / iqr
        return np.clip(normed, -5.0, 5.0)
    
    # --- Urgency term: tanh-based slack sensitivity, prioritizing negative slack (violation risk)
    # Uses adaptive tau_urg scaled to observed lateness magnitude, avoiding hard thresholds
    abs_slack = np.abs(slack)
    tau_urg = np.where(
        np.any(slack < 0),
        np.median(abs_slack[slack < 0]) + eps,
        np.median(abs_slack) + 0.1
    )
    urgency_raw = np.tanh(-slack / tau_urg)  # [-1,1]: higher when slack < 0; saturates smoothly
    urgency_norm = robust_normalize(urgency_raw)
    # Map to [0.05, 2.0] — strong monotonic priority boost for urgent tasks
    urgency_term = 0.05 + 1.95 * (urgency_raw + 1.0) / 2.0
    
    # --- Critical Path Importance (CP): gated by top-30% upward_rank, normalized by latency
    # Avoids over-weighting low-latency high-rank tasks; uses percentile gating (Parent 2)
    uprank_percentile = np.percentile(upward_rank, 70) if upward_rank.size > 1 else np.mean(upward_rank)
    cp_mask = (upward_rank >= uprank_percentile).astype(float)
    total_latency = min_exec_time + min_comm_time + eps
    cp_base = (upward_rank * remaining_work + eps) / (total_latency + eps)
    cp_gated = cp_base * cp_mask
    cp_norm = robust_normalize(cp_gated)
    # Smooth mapping to [0.2, 1.8] — preserves ordering, avoids zero weights
    cp_term = 0.2 + 1.6 * np.clip((cp_norm + 3.0) / 6.0, 0.0, 1.0)
    
    # --- Energy Efficiency Term: SEER-like (latency per energy), but with deadline-aware gating
    # Higher score = better efficiency; gated down when slack < 0 (deadline override)
    seer_base = total_latency / (min_incremental_energy + eps)  # higher → more efficient
    seer_gate = np.where(slack < 0, 0.25, 1.0)  # aggressive de-prioritization under violation
    # Uncertainty damping: reduce efficiency weight when uncertainty is high *and* slack is positive
    avg_unc = np.mean(uncertainty + eps)
    unc_damp = np.where(
        slack >= 0,
        np.clip(1.0 - 0.4 * (uncertainty / (avg_unc + eps)), 0.3, 1.0),
        1.0
    )
    seer_gated = seer_base * seer_gate * unc_damp
    seer_norm = robust_normalize(seer_gated)
    seer_term = 0.3 + 1.4 * np.clip((seer_norm + 3.0) / 6.0, 0.0, 1.0)
    
    # --- Fairness & Aging: linear wait-time boost *only* under deadline pressure
    # Proportional to sqrt(-slack) to penalize long waits when lateness looms (Parent 1 insight)
    lateness_margin = np.maximum(-slack + eps, 0.0)
    wait_ratio = ready_wait_time / (total_latency + eps)
    fairness_boost = np.where(
        slack < 0,
        np.clip(0.1 * wait_ratio * np.sqrt(lateness_margin), 0.0, 0.12),
        0.0
    )
    
    # --- Uncertainty Risk Amplification: continuous, smooth, and slack-conditioned
    # Inspired by Parent 1's sigmoid gate, but using tanh for numerical stability and bounded output
    risk_offset = np.tanh((1.0 - slack) / 0.5)  # ~1 when slack << 1, ~0 when slack >> 1, smooth transition
    unc_risk = uncertainty * risk_offset * 0.25
    unc_risk = np.clip(unc_risk, 0.0, 0.2)
    unc_norm = robust_normalize(unc_risk)
    unc_term = -0.1 * unc_norm  # small penalty: higher uncertainty → slightly lower priority (more cautious)
    
    # --- Composite score: multiplicative coupling (Parent 2 strength) + additive fairness & risk
    # Ensures urgency × CP × energy dominates; fairness adds offset; uncertainty applies fine-grained correction
    base_score = urgency_term * cp_term * seer_term
    score = base_score + fairness_boost + unc_term
    
    # Final sanitization: ensure finite, deterministic output with shape (N,)
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)
    
    # Enforce shape (N,) — critical for N=1 case
    return score.reshape(-1)
