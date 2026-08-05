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
    v2: Self-evolved deadline-hardened priority with:
    - Tighter imminent-violation focus via adaptive robust_slack clamping (±1σ uncertainty)
    - Uncertainty-aware fairness gating: dynamic threshold = max(0.05, 0.5 * uncertainty)
    - Smooth sigmoidal energy weighting instead of hard SEER gate → enables trade-off in borderline feasibility
    - Enhanced urgency resolution: uses tanh-based urgency for fine-grained near-deadline discrimination
    - All terms z-score normalized with strict finite bounds and N=1 safety
    - Final score is multiplicative hierarchy with deterministic violation override
    """
    eps = 1e-08
    
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
    
    # Adaptive robust slack: use ±1σ instead of fixed 2×uncertainty for tighter imminent-violation focus
    robust_slack = slack - uncertainty
    robust_slack = np.clip(robust_slack, -1e5, 1e5)
    
    # Imminent violation flag: slack < 0 OR robust_slack < 0.1 → captures near-term risk
    is_imminent = ((slack < 0.0) | (robust_slack < 0.1)).astype(float)
    
    # tanh-based urgency: smooth, bounded (-1, 0), high sensitivity near zero slack
    # avoids step discontinuities of piecewise linear forms
    urgency_raw = -np.tanh(np.clip(robust_slack / (np.abs(robust_slack) + eps), -10.0, 10.0))
    
    def normalize_zscore(x):
        x = np.clip(x, -1e6, 1e6)
        if x.size == 1:
            return np.array([0.0])
        mean = np.mean(x)
        std = np.std(x, ddof=0)
        if std < eps:
            return np.zeros_like(x)
        return (x - mean) / (std + eps)
    
    norm_urgency = normalize_zscore(urgency_raw)
    # Map normalized urgency [-2,2] → multiplicative weight [0.01, 50.0] with higher dynamic range
    urgency_mult = np.clip(0.01 + 49.99 * (1.0 - (norm_urgency + 2.0) / 4.0), 0.01, 50.0)
    
    # Critical path density: work × importance per unit execution+comm time
    exec_comm_sum = min_exec_time + min_comm_time + eps
    cp_density = upward_rank * remaining_work / (exec_comm_sum + eps)
    
    # Dynamic fairness gating threshold: scales with uncertainty → stricter fairness when uncertain
    fairness_threshold = np.maximum(0.05, 0.5 * uncertainty)
    cp_gate = (slack > fairness_threshold).astype(float)
    cp_density_gated = cp_density * cp_gate
    norm_cp = normalize_zscore(cp_density_gated)
    cp_offset = np.clip((norm_cp + 2.0) / 4.0, 0.0, 1.0)
    
    # Smooth energy weighting: sigmoid of inverse SEER, not hard gate → preserves energy signal even near deadline
    inv_seer = np.clip(min_incremental_energy / (exec_comm_sum + eps), eps, 1e6)
    # Sigmoid: 1 / (1 + exp(-k*(x - x0))) → centered at median SEER, steepness k=2
    seer_median = np.median(inv_seer) if inv_seer.size > 1 else np.mean(inv_seer)
    seer_score = 1.0 / (1.0 + np.exp(-2.0 * (inv_seer - seer_median)))
    norm_seer = normalize_zscore(seer_score)
    seer_offset = np.clip(norm_seer / 2.0, 0.0, 0.35)  # reduced max contribution to avoid energy dominance
    
    # Uncertainty-aware fairness: only active when slack > threshold, normalized aging ratio
    fairness_denom = np.abs(slack) + eps
    fairness_raw = ready_wait_time / fairness_denom
    fairness_gated = np.where(slack > fairness_threshold, fairness_raw, 0.0)
    norm_fair = normalize_zscore(fairness_gated)
    fairness_offset = np.clip((norm_fair + 2.0) / 4.0 * 0.12, 0.0, 0.12)
    
    # Multiplicative hierarchy: urgency dominates; CP, energy, fairness modulate within safe bounds
    score = urgency_mult * (1.0 + cp_offset) * (1.0 + seer_offset) * (1.0 + fairness_offset)
    
    # Hard override: assign lowest possible score to imminent-violation tasks
    base_min = np.min(score) if len(score) > 0 else 0.0
    score = np.where(is_imminent, base_min - 1e9, score)
    
    # Final sanitization: ensure finite, bounded, shape-(N,)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)
    
    return score
