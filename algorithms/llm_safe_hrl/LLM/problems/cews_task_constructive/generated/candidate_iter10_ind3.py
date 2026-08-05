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
    v2 mutation: Replaces adaptive normalization with robust MAD-based scaling;
    introduces *slack-gated energy efficiency ratio* (EER = 1/(energy * (1 + |slack|+eps)) for feasible tasks);
    replaces arctan urgency with stabilized inverse-slack urgency: 1/(|slack|+eps) clipped and sign-aware;
    adds *critical-path pressure* term: upward_rank * (1 + max(0, -slack)/median_positive_slack) to amplify priority under violation;
    uses multiplicative uncertainty damping only on latency (not energy) when 0 < slack <= median_slack;
    removes fairness clipping and instead uses linear wait-boost scaled by slack abundance threshold;
    all terms numerically sanitized, no division by zero, no NaN/inf propagation.
    """
    eps = 1e-08
    # Ensure float64 & sanitize inputs
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=0.0, posinf=eps, neginf=eps)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=0.0, posinf=eps, neginf=eps)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=eps, neginf=eps)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=eps, neginf=-eps)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=0.0, posinf=eps, neginf=0.0)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=0.0, posinf=eps, neginf=0.0)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=eps, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=0.0, posinf=eps, neginf=0.0)

    N = len(slack)
    if N == 0:
        return np.array([], dtype=float)

    # Robust MAD-based normalization (more stable than IQR for small N or flat arrays)
    def normalize_mad(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if N == 1:
            return np.array([0.0])
        med = np.median(x)
        abs_dev = np.abs(x - med)
        mad = np.median(abs_dev)
        scale = mad if mad > eps else np.max(np.abs(x - med)) if np.any(x != med) else eps
        return (x - med) / (scale + eps)

    # === Urgency: Inverse-slack with sign-aware saturation ===
    # For negative slack: strong linear penalty → high priority (low score)
    # For positive slack: decaying priority as slack grows → use 1/(|slack|+eps), bounded above by 1/eps_max
    eps_max = 1000.0
    inv_slack_raw = np.where(slack < 0, -slack + eps, eps / (np.abs(slack) + eps))
    inv_slack_raw = np.clip(inv_slack_raw, 0.0, eps_max)
    norm_urgency = normalize_mad(inv_slack_raw)
    urgency_term = -3.0 * norm_urgency  # Higher magnitude for strict DDL enforcement

    # === Critical-path pressure: upward_rank amplified under deadline stress ===
    # Only activate amplification when slack <= 0; otherwise base weight = 1.0
    positive_slack_mask = slack > 0
    median_pos_slack = np.median(slack[positive_slack_mask]) if np.any(positive_slack_mask) else 1.0
    slack_pressure = np.where(slack <= 0, 1.0 + (-slack) / (median_pos_slack + eps), 1.0)
    cp_pressure_base = upward_rank * slack_pressure
    norm_cp = normalize_mad(cp_pressure_base)
    cp_term = -1.5 * norm_cp

    # === Slack-gated Energy Efficiency Ratio (EER): prioritize low-energy *only* when feasible ===
    # EER = 1 / (energy * (1 + |slack|+eps)) → higher EER = better efficiency under margin
    # When slack < 0: set EER = 0 → ignore energy; DDL safety dominates
    eer_base = 1.0 / (min_incremental_energy * (1.0 + np.abs(slack) + eps) + eps)
    eer_masked = np.where(slack >= 0, eer_base, 0.0)
    norm_eer = normalize_mad(eer_masked)
    eer_term = -1.0 * norm_eer

    # === Latency penalty: execution + comm, modulated by uncertainty only in tight-but-feasible zone ===
    base_latency = min_exec_time + min_comm_time + eps
    # Apply uncertainty factor only when 0 < slack <= median_pos_slack (tight feasible window)
    tight_feasible_mask = (slack > 0) & (slack <= median_pos_slack)
    uncertainty_factor = np.where(tight_feasible_mask, 1.0 + uncertainty / (np.max(uncertainty) + eps), 1.0)
    latency_penalty = base_latency * uncertainty_factor
    norm_latency = normalize_mad(latency_penalty)
    latency_term = 0.7 * norm_latency

    # === Fairness: linear wait-boost, activated only when slack > median_pos_slack (abundant margin) ===
    # Scales linearly with wait time, but capped by slack abundance: boost ∝ wait_time / (slack + eps)
    wait_boost_raw = np.where(slack > median_pos_slack, 
                              ready_wait_time / (slack + eps), 
                              0.0)
    norm_wait = normalize_mad(wait_boost_raw)
    wait_term = -0.15 * norm_wait

    # Assemble final score: lower = better
    score = urgency_term + cp_term + eer_term + latency_term + wait_term

    # Final sanitization: ensure finite output
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)

    return score
