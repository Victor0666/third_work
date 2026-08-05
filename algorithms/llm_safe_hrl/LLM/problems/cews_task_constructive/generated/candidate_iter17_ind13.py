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
    v2: Hybrid safety-aware, energy-efficient, and critical-path-focused priority.
    Combines Parent 2's slack-gated SEER & sharp urgency with Parent 1's robust slack margin,
    adaptive uncertainty amplification, and lexicographic violation override.
    Key novelties:
      - Dual-safety slack: uses robust_slack = slack - 2*uncertainty for violation detection AND gating,
      - Unified urgency: tanh(-robust_slack / tau_urg) — sharper & risk-aware near deadline,
      - Critical-path density (CPD) enhanced with remaining_work normalization to avoid bias toward leaf tasks,
      - SEER activated only when robust_slack > median_positive_robust_slack (not just >0), improving headroom sensitivity,
      - Fairness as *risk-suppressed relative aging*: sqrt(ready_wait_time) / (1 + max(0, -robust_slack)), bounded and starvation-safe,
      - All normalizations use MAD with N=1 guard and strict finite clipping [-3.0, 3.0],
      - Violation override sets score = -inf (smallest possible) for any task with slack < 0 — absolute priority.
    """
    eps = 1e-08

    # Sanitize inputs: convert, replace NaN/inf with safe finite values
    def sanitize(x):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(x, nan=eps, posinf=1e6, neginf=1e-6)

    min_exec_time = sanitize(min_exec_time)
    min_comm_time = sanitize(min_comm_time)
    min_incremental_energy = sanitize(min_incremental_energy)
    slack = sanitize(slack)
    upward_rank = sanitize(upward_rank)
    remaining_work = sanitize(remaining_work)
    ready_wait_time = sanitize(ready_wait_time)
    uncertainty = sanitize(uncertainty)

    # Robust slack: pessimistic estimate accounting for uncertainty risk
    robust_slack = np.clip(slack - 2.0 * uncertainty, -1e6, 1e6)

    # Absolute violation override: any task already violating deadline gets top priority
    violation_mask = slack < 0
    base_score = np.full_like(slack, 0.0)

    # Compute median of positive robust_slack for adaptive gating (fallback if none exist)
    positive_robust = robust_slack[robust_slack > 0]
    median_pos_robust = np.median(positive_robust) if positive_robust.size > 0 else 1.0
    median_pos_robust = max(median_pos_robust, eps)

    # --- Urgency term: sharp, risk-aware, and normalized ---
    tau_urg = 0.4
    urgency_raw = np.tanh(-robust_slack / tau_urg)  # high urgency when robust_slack is negative/small
    # Normalize robustly
    def robust_mad_normalize(x):
        x = np.clip(x, -1e6, 1e6)
        if x.size == 1:
            return np.array([0.0])
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + eps
        normed = (x - med) / mad
        return np.clip(normed, -3.0, 3.0)
    urgency_norm = robust_mad_normalize(urgency_raw)
    urgency_term = -4.0 * urgency_norm

    # --- Critical-Path Density (CPD): importance per latency cost, scaled by work relevance ---
    base_latency = min_exec_time + min_comm_time + eps
    # CPD = (upward_rank / base_latency) * (remaining_work / (median_remaining_work + eps))
    median_remaining = np.median(remaining_work) + eps
    cpd_base = (upward_rank / base_latency) * (remaining_work / median_remaining)
    # Gate CPD by robust_slack: suppress when deadline pressure is high
    cpd_gate = np.tanh(np.maximum(robust_slack, 0.0) / tau_urg)
    cpd_masked = cpd_base * cpd_gate
    cpd_norm = robust_mad_normalize(cpd_masked)
    cpd_term = -2.0 * cpd_norm

    # --- Slack-Gated SEER (Slack-activated Energy Efficiency Ratio) ---
    seer_base = base_latency / (min_incremental_energy + eps)
    # Activate only when robust slack exceeds typical safe headroom
    seer_activation = np.where(robust_slack > median_pos_robust,
                               np.tanh((robust_slack - median_pos_robust) / (median_pos_robust + eps)),
                               0.0)
    seer_masked = seer_base * seer_activation
    seer_norm = robust_mad_normalize(seer_masked)
    seer_term = -1.2 * seer_norm

    # --- Risk-adjusted fairness: suppresses starvation only when robust_slack >= 0 ---
    wait_safe = np.maximum(ready_wait_time, 0.0) + eps
    sqrt_wait = np.sqrt(wait_safe)
    # Denominator grows linearly with lateness risk; prevents explosion when robust_slack < 0
    fairness_denom = 1.0 + np.maximum(-robust_slack, 0.0)
    fairness_raw = sqrt_wait / fairness_denom
    fairness_clipped = np.clip(fairness_raw, 0.0, 0.5)
    fairness_norm = robust_mad_normalize(fairness_clipped)
    fairness_term = -0.1 * fairness_norm

    # Combine terms: urgency dominates, then CPD, then SEER, then fairness
    score = urgency_term + cpd_term + seer_term + fairness_term

    # Apply violation override: smallest possible score for violated tasks
    score = np.where(violation_mask, -1e12, score)

    # Final sanitization: ensure finite, deterministic output
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)

    return score
