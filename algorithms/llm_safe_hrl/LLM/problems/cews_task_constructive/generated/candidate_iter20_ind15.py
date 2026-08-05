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
    # Robust sanitization: convert to float, handle NaN/inf/neginf safely
    eps = 1e-08
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

    # Robust normalization preserving magnitude semantics (v2-style improved)
    def robust_normalize(x):
        if x.size == 0:
            return np.zeros_like(x)
        if x.size == 1:
            return np.zeros_like(x)
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + eps
        normed = (x - med) / mad
        # Clip extreme outliers but preserve relative ordering better than hard [-2,2]
        return np.clip(normed, -3.0, 3.0)

    # --- Urgency term: strict deadline dominance with robust slack handling ---
    # Use uncertainty-aware slack: conservative margin for risk-aware deadline adherence
    robust_slack = slack - 1.5 * uncertainty
    # Tanh-based urgency: smooth, bounded, monotonic; high priority for negative slack
    tau_urg = max(0.1, np.percentile(np.abs(robust_slack), 75) + eps)
    urgency_raw = np.tanh(-robust_slack / tau_urg)
    urgency_norm = robust_normalize(urgency_raw)
    # Map to multiplicative factor [0.1, 2.0] — zero-slack gives ~1.0, high urgency >1.0
    urgency_term = 0.1 + 1.9 * (urgency_raw + 1.0) / 2.0

    # --- Critical Path Density Ratio (CPDR): importance per latency cost ---
    total_latency = min_exec_time + min_comm_time + eps
    # Gating: only activate CPDR for tasks above median criticality (reduces noise)
    uprank_med = np.median(upward_rank) if upward_rank.size > 1 else np.mean(upward_rank)
    cpdr_mask = (upward_rank >= uprank_med * 0.4).astype(float)
    cpdr_base = (upward_rank + eps) * (remaining_work + eps) / (total_latency + eps)
    cpdr_gated = cpdr_base * cpdr_mask
    cpdr_norm = robust_normalize(cpdr_gated)
    # Multiplicative term: centered at 1.0, bounded to prevent explosion
    cpdr_term = 1.0 + 0.8 * np.clip(cpdr_norm, -1.0, 1.0)

    # --- Sustainable Energy Efficiency Ratio (SEER): energy efficiency gated by urgency ---
    # Higher SEER = better energy efficiency per latency unit → prefer when safe
    seer_base = total_latency / (min_incremental_energy + eps)
    # Gate: suppress SEER when slack is critically negative (urgency dominates)
    seer_gate = np.exp(-np.maximum(0.0, -robust_slack) / (tau_urg + eps))
    # Uncertainty dampening: reduce confidence in SEER when uncertainty is high and slack > 0
    unc_damp = np.where(
        robust_slack > 0,
        np.clip(1.0 - (uncertainty / (np.mean(uncertainty + eps) + eps)), 0.3, 1.0),
        1.0
    )
    seer_gated = seer_base * seer_gate * unc_damp
    seer_norm = robust_normalize(seer_gated)
    seer_term = 1.0 + 0.7 * np.clip(seer_norm, -1.0, 1.0)

    # --- Fairness offset: bounded additive boost *only* for violated or near-violated tasks ---
    # Prevents starvation without distorting multiplicative hierarchy
    fairness_boost = np.where(
        robust_slack < 0.05,  # Include near-deadline tasks to avoid last-moment surprises
        np.clip(
            0.08 * (ready_wait_time / (total_latency + eps)) * 
            np.sqrt(np.maximum(-robust_slack + eps, 0.0)),
            0.0, 0.12
        ),
        0.0
    )
    fairness_norm = robust_normalize(fairness_boost)
    fairness_offset = np.clip(fairness_norm, 0.0, 0.12)

    # --- Composite score: multiplicative hierarchy + bounded fairness offset ---
    # Preserves lexicographic dominance: urgency > CPDR > SEER
    base_score = urgency_term * cpdr_term * seer_term
    score = base_score + fairness_offset

    # Final sanitization: ensure finite, deterministic, shape-(N,) output
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)
    return score.reshape(-1)
