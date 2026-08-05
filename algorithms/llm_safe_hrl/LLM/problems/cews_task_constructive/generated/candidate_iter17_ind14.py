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
    v2: Deadline-hardened, energy-efficient, risk-aware priority with deterministic monotonic normalization.
    
    Key innovations:
    - Dual-mode urgency: strict linear penalty for slack <= 0 (hard DDL enforcement), smooth tanh for positive slack
    - Physics-informed τ (1.0s urgency, 3.0s energy horizon) — stable and interpretable
    - Robust SEER (energy per latency) as primary driver when deadlines are safe (slack > 0)
    - Critical path density gated by slack > τ_urgency *only* when urgency is low — avoids over-prioritizing idle critical tasks
    - Fairness via wait-boost scaled by normalized slack surplus (not clipped ratio) for smoother starvation prevention
    - Uncertainty boost activated only under clear safety margin (slack > 2.0 AND uncertainty > 0.15) — no fragile scaling
    - Deterministic z-score normalization with std fallback and bounded clipping for monotonicity & stability
    - All NaN/inf/zero safeguards applied pre- and post-normalization; final score strictly finite and deterministic
    """
    eps = 1e-08
    # Safe cast and nan/inf handling for all inputs
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=0.0, posinf=1e6, neginf=0.0)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=0.0, posinf=1e6, neginf=0.0)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=1e6, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=0.0, posinf=1e6, neginf=0.0)

    def normalize_deterministic(x):
        x = np.clip(x, -1e6, 1e6)
        if x.size == 1:
            return np.array([0.0])
        mean = np.mean(x)
        std = np.std(x, ddof=0)
        if std < eps:
            return np.zeros_like(x)
        return np.clip((x - mean) / (std + eps), -5.0, 5.0)

    tau_urgency = 1.0   # sub-second urgency threshold
    tau_energy = 3.0    # empirical energy discount horizon

    # === URGENCY TERM: Hard deadline enforcement ===
    # Linear penalty for violated or imminent deadlines (slack <= 0), else smooth tanh decay
    urgency_raw = np.where(slack <= 0, -slack * 4.0, np.tanh(-slack / tau_urgency))
    urgency_mask = (slack <= 0).astype(float)  # full weight on hard violation
    norm_urgency = normalize_deterministic(urgency_raw * urgency_mask)
    urgency_term = -6.0 * norm_urgency

    # === ENERGY EFFICIENCY TERM (SEER): Dominant when slack > 0 ===
    exec_comm_sum = min_exec_time + min_comm_time + eps
    seer_base = np.clip(min_incremental_energy / exec_comm_sum, 0.0, 1e5)
    # Energy preference decays exponentially with slack surplus, but only when safe
    slack_factor = np.where(slack > 0, np.exp(-np.clip(slack, 0.0, 10.0) / tau_energy), 0.0)
    seer_masked = seer_base * slack_factor
    norm_seer = normalize_deterministic(seer_masked)
    seer_term = -2.5 * norm_seer

    # === CRITICAL PATH DENSITY: Only relevant when not urgent ===
    cp_density = upward_rank * remaining_work / (exec_comm_sum + eps)
    cp_gate = (slack > tau_urgency).astype(float)  # activate only when clearly safe
    cp_density_gated = cp_density * cp_gate
    norm_cp_density = normalize_deterministic(cp_density_gated)
    cp_density_term = 0.6 * norm_cp_density

    # === FAIRNESS TERM: Wait-boost scaled by normalized slack surplus ===
    # Avoids starvation: longer wait matters more when deadline is comfortably met
    slack_surplus_norm = np.clip((slack - tau_urgency) / (tau_energy + eps), 0.0, 1.0)
    wait_boost = np.clip(ready_wait_time / (exec_comm_sum + eps), 0.0, 10.0) * slack_surplus_norm
    norm_wait = normalize_deterministic(wait_boost)
    fairness_term = -0.3 * norm_wait

    # === UNCERTAINTY RISK TERM: Conservative activation ===
    unc_boost = np.where((slack > 2.0) & (uncertainty > 0.15), 
                        np.clip(uncertainty * 0.4, 0.0, 0.2), 0.0)
    norm_unc = normalize_deterministic(unc_boost)
    unc_term = -0.15 * norm_unc

    # Combine all terms — urgency dominates violations, SEER dominates safe region
    score = urgency_term + seer_term + cp_density_term + fairness_term + unc_term

    # Final sanitization: ensure finite, deterministic, shape-(N,)
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e8, 1e8)
    return score.reshape(-1)
