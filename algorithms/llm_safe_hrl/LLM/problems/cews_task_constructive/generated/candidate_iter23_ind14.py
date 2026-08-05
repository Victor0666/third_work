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
    v2: Hybrid deadline-hardened, energy-aware priority with:
    - Multiplicative urgency dominance for strict deadline enforcement
    - Robust uncertainty-aware slack clamping and gating
    - Physics-aligned inverse SEER (energy per unit critical work)
    - Starvation-robust fairness gated by slack feasibility
    - Z-score normalization with fallback for N=1 and low-variance cases
    - Finite-bounded, deterministic, and numerically stable output
    """
    eps = 1e-08
    
    # Sanitize all inputs: convert to float, replace NaN/inf with safe values
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
    
    # Robust slack: subtract uncertainty margin, clamp to avoid overflow
    robust_slack = slack - 2.0 * uncertainty
    robust_slack = np.clip(robust_slack, -1e5, 1e5)
    
    # Hard violation mask: tasks with slack < 0 get highest priority (lowest score)
    is_violated = (slack < 0.0).astype(float)
    
    # Urgency: monotonic, bounded mapping: -1.0 (critical) → 0.0 (safe), using clipped rational
    # Avoids tanh overflow; smooth, invertible, and zero-at-safe boundary
    urgency_raw = np.where(
        robust_slack <= 0.0,
        -1.0,
        -robust_slack / (np.abs(robust_slack) + 1.0)
    )
    
    # Normalize urgency via z-score with variance fallback
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
    # Map normalized urgency [-2,2] → multiplicative factor [0.05, 20.0] with strong penalty near -1
    urgency_mult = np.clip(0.05 + 19.95 * (1.0 - (norm_urgency + 2.0) / 4.0), 0.05, 20.0)
    
    # Critical path density: importance × remaining work per time budget
    exec_comm_sum = min_exec_time + min_comm_time + eps
    cp_density = upward_rank * remaining_work / (exec_comm_sum + eps)
    
    # Gate CP term only when slack > 0.5 (feasible regime), else zero to avoid misleading bias
    cp_gate = (slack > 0.5).astype(float)
    cp_density_gated = cp_density * cp_gate
    norm_cp = normalize_zscore(cp_density_gated)
    # Offset: map [-2,2] → [0.0, 1.0] for multiplicative contribution
    cp_offset = np.clip((norm_cp + 2.0) / 4.0, 0.0, 1.0)
    
    # Inverse SEER (Energy Efficiency Ratio): energy cost per critical-path-relevant work/time
    # Only active when slack > 2.0 AND uncertainty low → high-confidence energy optimization
    inv_seer = np.clip(min_incremental_energy / (exec_comm_sum + eps), eps, 1e6)
    energy_gate = ((slack > 2.0) & (uncertainty < 0.25)).astype(float)
    seer_masked = inv_seer * energy_gate
    norm_seer = normalize_zscore(seer_masked)
    # Smooth offset: [-2,2] → [0.0, 0.4] — modest energy-aware boost when safe
    seer_offset = np.clip(norm_seer / 2.0, 0.0, 0.4)
    
    # Fairness: aging ratio, but *only* activated under slack > 0.1 to prevent starvation in feasible region
    fairness_denom = np.abs(slack) + eps
    fairness_raw = ready_wait_time / fairness_denom
    fairness_gated = np.where(slack > 0.1, fairness_raw, 0.0)
    norm_fair = normalize_zscore(fairness_gated)
    fairness_offset = np.clip((norm_fair + 2.0) / 4.0 * 0.15, 0.0, 0.15)
    
    # Core multiplicative priority: urgency dominates; others provide controlled refinement
    score = urgency_mult * (1.0 + cp_offset) * (1.0 + seer_offset) * (1.0 + fairness_offset)
    
    # Override violated tasks with minimum possible score (highest priority)
    base_min = np.min(score) if len(score) > 0 else 0.0
    score = np.where(is_violated, base_min - 1e9, score)
    
    # Final sanitization: finite bounds, no NaN/inf
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)
    
    return score
