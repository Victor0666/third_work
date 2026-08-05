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
    v2 hybrid priority rule: combines Parent 2's lexicographic urgency dominance and robust MAD normalization
    with Parent 1's hard-gated urgency penalty, asymmetric uncertainty window, and aging boost.
    Key improvements:
      - Strict deadline enforcement: negative slack triggers *dominant* penalty (not just linear)
      - Urgency term uses hard-gated sigmoid + direct violation bonus for slack < 0
      - Energy efficiency only activated when slack >= 0 AND upward_rank > 0 (avoids leaf noise)
      - Uncertainty applied asymmetrically: active for |slack| <= 1.5 with quadratic weighting near zero
      - Aging boost added for tasks in tight-but-feasible zone (slight positive slack + waiting)
      - All terms normalized via robust MAD with degenerate-case fallback (N=1 or constant arrays)
      - Lexicographic weight scaling: urgency (5.0) >> energy (-1.8) >> uncertainty (0.25) >> aging (0.12)
    """
    eps = 1e-08
    # Sanitize inputs to finite values
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=0.0, posinf=1e6, neginf=0.0)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=0.0, posinf=1e6, neginf=0.0)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=1e6, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=0.0, posinf=1e6, neginf=0.0)

    def robust_mad_normalize(x):
        """MAD-based normalization with safe fallback for degenerate cases (N=1 or constant)."""
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 0:
            return np.zeros_like(x)
        if x.size == 1:
            return np.zeros_like(x)
        median_x = np.median(x)
        mad = np.median(np.abs(x - median_x))
        # Fallback to range if MAD is too small (flat array)
        scale = mad if mad > eps else (np.max(x) - np.min(x)) + eps
        scale = max(scale, eps)
        normed = (x - median_x) / scale
        return np.clip(normed, -3.0, 3.0)

    # === URGENCY TERM (dominant: ensures DDL compliance) ===
    # Hard-gated: strong penalty for slack < 0, smooth urgency for slack > 0
    slack_abs = np.abs(slack)
    # Direct violation bonus: large negative score for any slack < 0
    violation_bonus = np.where(slack < 0.0, -1000.0, 0.0)
    # Smooth urgency for feasible region: sigmoid centered at slack=0
    sigmoid_urgency = 1.0 / (1.0 + np.exp(-slack / (np.abs(np.median(slack)) + 0.1)))
    raw_urgency = violation_bonus + sigmoid_urgency
    norm_urgency = robust_mad_normalize(raw_urgency)
    urgency_term = 5.0 * norm_urgency  # highest weight: urgency dominates

    # === ENERGY-EFFICIENCY TERM (active only when deadlines are feasible) ===
    # Energy-normalized critical work: remaining_work / energy, gated by feasibility
    ecw_base = remaining_work / (min_incremental_energy + eps)
    ecw_masked = np.where((slack >= 0.0) & (upward_rank > eps), ecw_base, 0.0)
    norm_ecw = robust_mad_normalize(ecw_masked)
    ecw_term = -1.8 * norm_ecw  # negative weight: higher ECW = better energy efficiency

    # === UNCERTAINTY TERM (asymmetric risk window around slack=0) ===
    # Active for |slack| <= 1.5, with strongest impact near zero slack (quadratic weighting)
    slack_abs_clipped = np.clip(slack_abs, 0.0, 1.5)
    unc_weight = 1.0 - (slack_abs_clipped / (1.5 + eps)) ** 2
    uncertainty_active = (slack_abs <= 1.5) & (uncertainty > 0.02)
    uncertainty_penalty = np.where(uncertainty_active, uncertainty * unc_weight, 0.0)
    norm_unc = robust_mad_normalize(uncertainty_penalty)
    uncertainty_term = 0.25 * norm_unc  # positive: higher uncertainty = lower priority

    # === AGING BOOST (for tasks waiting in tight-but-feasible zone) ===
    # Active when slack is slightly positive (0 < slack <= 0.5) and task has waited
    aging_cond = (slack > 0.0) & (slack <= 0.5) & (ready_wait_time > 0.03)
    aging_boost = np.where(aging_cond, np.clip(ready_wait_time / (0.3 + slack), 0.0, 1.0), 0.0)
    aging_term = 0.12 * aging_boost  # small positive boost: encourages timely scheduling of borderline tasks

    # Combine all terms with lexicographic emphasis
    score = urgency_term + ecw_term + uncertainty_term + aging_term

    # Final sanitization
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)

    return score
