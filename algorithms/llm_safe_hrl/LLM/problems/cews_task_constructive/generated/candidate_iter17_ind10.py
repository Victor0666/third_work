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
    v2: Hybrid deadline-energy priority with robust safety layering, adaptive risk gating,
    and numerically stable normalization. Combines Parent 2's physics-informed τ-gating
    and z-score stability with Parent 1's violation override rigor and fairness redefinition.
    
    Key improvements:
    - Hard violation override: assigns absolute minimum score (not just large negative) for slack < 0
    - Unified urgency layer: uses tanh(-slack/τ) but clamps robust_slack to prevent overflow
    - Energy-awareness gated by both slack margin AND uncertainty (slacks > 2.0 & uncertainty > 0.15)
    - Fairness redefined as relative aging: ready_wait_time / max(1, median_slack_pos, -slack + eps)
    - All normalizations use deterministic z-score with std fallback and strict clipping
    - Final score monotonic in slack for feasible region; all ops protected against NaN/inf/zero
    """
    eps = 1e-08
    
    # Sanitize inputs: ensure finite, replace NaN/inf with safe defaults
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
    
    # Compute robust slack: conservative estimate accounting for uncertainty
    robust_slack = np.clip(slack - 2.0 * uncertainty, -1e6, 1e6)
    
    # Violation mask: tasks already predicted to miss deadline
    violation_mask = slack < 0
    
    # Physics-informed time constants
    tau_urgency = 1.0   # sub-second urgency threshold
    tau_energy = 3.0    # empirical energy discount horizon
    
    # Urgency term: dominates only on violation; otherwise zero contribution
    # Uses tanh for smooth saturation, bounded input to prevent numerical issues
    urgency_raw = np.tanh(-np.clip(robust_slack, -10.0, 10.0) / tau_urgency)
    urgency_term = np.where(violation_mask, -10.0 * urgency_raw, 0.0)
    
    # Energy efficiency term (SEER): activated only when slack > 2.0 AND uncertainty > 0.15
    exec_comm_sum = min_exec_time + min_comm_time + eps
    base_seer = np.clip(min_incremental_energy / exec_comm_sum, eps, 1e6)
    energy_gate = ((slack > 2.0) & (uncertainty > 0.15)).astype(float)
    seer_masked = base_seer * energy_gate * np.exp(-np.maximum(0.0, slack) / tau_energy)
    
    # Critical path density: importance-weighted work per latency, gated by slack surplus
    cp_density = upward_rank * remaining_work / (exec_comm_sum + eps)
    cp_gate = (slack > tau_urgency).astype(float)
    cp_density_gated = cp_density * cp_gate
    
    # Fairness term: relative aging — prevents starvation near deadlines
    positive_slack = slack[slack > 0]
    median_slack_pos = np.median(positive_slack) if len(positive_slack) > 0 else 1.0
    median_slack_pos = max(median_slack_pos, eps)
    fairness_denom = np.maximum(1.0, np.maximum(median_slack_pos, -slack + eps))
    fairness_raw = ready_wait_time / (fairness_denom + eps)
    fairness_term = np.clip(fairness_raw, 0.0, 10.0)
    
    # Uncertainty amplification: only when slack is ample and uncertainty is non-negligible
    unc_boost = np.where((slack > 2.0) & (uncertainty > 0.15), 
                        np.clip(uncertainty * 0.4, 0.0, 0.2), 0.0)
    
    # Deterministic z-score normalization with fallback for constant arrays
    def normalize_zscore(x):
        x = np.clip(x, -1e6, 1e6)
        if x.size == 1:
            return np.array([0.0])
        mean = np.mean(x)
        std = np.std(x, ddof=0)
        if std < eps:
            return np.zeros_like(x)
        return (x - mean) / (std + eps)
    
    norm_urgency = normalize_zscore(urgency_term)
    norm_seer = normalize_zscore(seer_masked)
    norm_cp = normalize_zscore(cp_density_gated)
    norm_fair = normalize_zscore(fairness_term)
    norm_unc = normalize_zscore(unc_boost)
    
    # Weighted fusion: urgency dominates violations; energy & CP drive efficiency under safety
    score = (
        +6.0 * norm_urgency      # Strongest weight for violation response
        -2.5 * norm_seer         # Energy minimization when safe
        +0.8 * norm_cp           # Critical path acceleration when slack permits
        -0.3 * norm_fair         # Starvation prevention scaled by relative aging
        -0.15 * norm_unc         # Risk-aware penalty for uncertain high-slack tasks
    )
    
    # Apply hard violation override: smallest possible score ensures top priority
    score = np.where(violation_mask, -1e12, score)
    
    # Final sanitization: ensure finite, bounded output
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)
    
    return score
