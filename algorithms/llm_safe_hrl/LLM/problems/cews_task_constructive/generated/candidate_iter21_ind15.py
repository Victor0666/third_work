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
    
    # Sanitize inputs: ensure finite, replace NaN/inf with safe values
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
    
    # Robust normalization preserving outlier sensitivity: use IQR-based scaling with fallback
    def robust_normalize(x):
        if x.size == 0:
            return np.zeros_like(x)
        if x.size == 1:
            return np.zeros_like(x)
        q25, q75 = np.percentile(x, [25, 75])
        iqr = q75 - q25 + eps
        med = np.median(x)
        normed = (x - med) / iqr
        # Clip only extreme outliers; retain tail signal for urgency
        return np.clip(normed, -5.0, 5.0)
    
    # Critical correction: avoid slack over-dampening — use *additive* uncertainty margin only for violation detection,
    # not for distorting urgency magnitude. Preserve raw slack for tanh scaling.
    # Define "violation zone" as slack < -uncertainty (i.e., predicted lateness exceeds uncertainty bound)
    violation_mask = (slack < -uncertainty).astype(float)
    
    # Urgency term: sharp, bounded, and slack-dominated. Use raw slack with adaptive tau to preserve outliers.
    # tau set to median absolute slack in violation zone, or global median if no violation
    abs_slack = np.abs(slack)
    tau_urg = np.median(abs_slack[slack < 0]) if np.any(slack < 0) else np.median(abs_slack) + eps
    tau_urg = max(tau_urg, 0.1)
    urgency_raw = np.tanh(-slack / tau_urg)  # [-1,1]; high priority when slack negative & large magnitude
    urgency_norm = robust_normalize(urgency_raw)
    # Map to [0.05, 2.0] — strict monotonic priority boost for urgency, no inversion
    urgency_term = 0.05 + 1.95 * (urgency_raw + 1.0) / 2.0
    
    # Critical-path delay reduction (CPDR): now weighted by remaining work AND upward rank,
    # but gated *only* when task is on critical path (upward_rank > top 30% percentile)
    total_latency = min_exec_time + min_comm_time + eps
    uprank_percentile = np.percentile(upward_rank, 70) if upward_rank.size > 1 else np.mean(upward_rank)
    cpdr_mask = (upward_rank >= uprank_percentile).astype(float)
    cpdr_base = (upward_rank * remaining_work + eps) / (total_latency + eps)
    cpdr_gated = cpdr_base * cpdr_mask
    cpdr_norm = robust_normalize(cpdr_gated)
    # Bounded linear scaling: [0.2, 1.8] to ensure CPDR never dominates urgency lexicographically
    cpdr_term = 0.2 + 1.6 * np.clip((cpdr_norm + 3.0) / 6.0, 0.0, 1.0)
    
    # SEER (System Energy Efficiency Ratio): latency-over-energy, dampened *only* under deadline risk
    seer_base = total_latency / (min_incremental_energy + eps)
    # Strong gate: suppress SEER only when violation imminent (slack < 0), otherwise full weight
    seer_gate = np.where(slack < 0, 0.3, 1.0)  # Aggressively deprioritize energy when deadline at risk
    unc_damp = np.where(slack >= 0, 
                        np.clip(1.0 - 0.5 * (uncertainty / (np.mean(uncertainty + eps) + eps)), 0.4, 1.0),
                        1.0)
    seer_gated = seer_base * seer_gate * unc_damp
    seer_norm = robust_normalize(seer_gated)
    seer_term = 0.3 + 1.4 * np.clip((seer_norm + 3.0) / 6.0, 0.0, 1.0)
    
    # Fairness: starvation avoidance *only* under violation, additive and strictly bounded (≤0.08)
    # Uses normalized wait-time per latency unit, scaled by violation severity
    fairness_boost = np.where(
        slack < 0,
        np.clip(
            0.08 * (ready_wait_time / (total_latency + eps)) * 
            np.sqrt(np.maximum(-slack + eps, 0.0)),
            0.0, 0.08
        ),
        0.0
    )
    # No normalization — keep fairness as small bounded offset to avoid hierarchy distortion
    fairness_offset = fairness_boost
    
    # Lexicographic multiplicative core: urgency × CPDR × SEER ensures deadline dominates
    base_score = urgency_term * cpdr_term * seer_term
    score = base_score + fairness_offset
    
    # Final sanitization: ensure finite, deterministic, shape-(N,)
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)
    return score.reshape(-1)
