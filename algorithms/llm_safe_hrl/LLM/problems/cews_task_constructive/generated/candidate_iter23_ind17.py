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
    # Deterministic, finite, and robust priority scoring for deadline-hardened energy-aware scheduling
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
    
    # --- HARD DEADLINE OVERRIDE (lexicographic dominance) ---
    # Assign -inf to any task already violating deadline (slack < 0), ensuring absolute priority
    violation_mask = slack < 0
    # Pre-apply override: tasks with slack < 0 get minimal score before composition
    base_score = np.full_like(slack, 1.0, dtype=float)
    
    # --- URGENCY: arctan-based, risk-adjusted, bounded [-1, 1] ---
    # Use robust_slack = slack - 3*uncertainty for stricter deadline safety under uncertainty
    robust_slack = slack - 3.0 * uncertainty
    robust_slack = np.clip(robust_slack, -1e5, 1e5)
    # Smooth, bounded urgency: arctan maps to [-π/2, π/2], normalize to [-1, 1]
    urgency_raw = np.arctan(robust_slack / (np.abs(robust_slack) + eps)) / (np.pi / 2)
    # Invert so negative robust_slack → high urgency → low score; clamp to avoid extremes
    urgency_score = np.clip(1.0 - urgency_raw, 0.05, 10.0)
    
    # --- CRITICALITY: MAD-normalized critical path density ---
    # Density = (upward_rank * remaining_work) / (exec + comm + eps) → higher = more critical
    exec_comm_sum = min_exec_time + min_comm_time + eps
    cp_density = upward_rank * remaining_work / (exec_comm_sum + eps)
    cp_density = np.clip(cp_density, 0.0, 1e6)
    
    def normalize_mad(x):
        x = np.clip(x, -1e6, 1e6)
        if x.size == 1:
            return np.array([0.0])
        median = np.median(x)
        mad = np.median(np.abs(x - median))
        if mad < eps:
            return np.zeros_like(x)
        return (x - median) / (mad + eps)
    
    norm_cp = normalize_mad(cp_density)
    # Sigmoid gating: map normalized criticality to [0.5, 1.5] multiplicative offset
    cp_offset = 0.5 + 1.0 / (1.0 + np.exp(-norm_cp))
    
    # --- ENERGY-EFFICIENCY: SEER-based (exec+comm)/energy → higher is better → invert for priority
    seer = exec_comm_sum / (min_incremental_energy + eps)
    seer_arctan = np.arctan(-seer) / (np.pi / 2)  # maps to [-1, 1], negative → high SEER → low score
    seer_offset = 0.5 + 0.5 * (1.0 + seer_arctan)  # maps to [0.0, 1.0] → then shift to [0.5, 1.5]
    
    # --- FAIRNESS: aging only when safe (robust_slack > 0), decay near deadline
    fairness_raw = ready_wait_time / (np.abs(slack) + 1.0)
    fairness_gated = np.where(robust_slack > 0.0, fairness_raw, 0.0)
    norm_fair = normalize_mad(fairness_gated)
    fairness_offset = 0.5 + 0.5 / (1.0 + np.exp(-norm_fair))
    
    # --- COMBINATION: multiplicative gating preserves hierarchical signal dominance ---
    # All terms in [0.5, 1.5] range → product stays well-bounded; no additive cancellation
    score = urgency_score * cp_offset * seer_offset * fairness_offset
    
    # Apply hard violation override *before* final clipping to guarantee lexicographic DDL priority
    score = np.where(violation_mask, -np.inf, score)
    
    # Final sanitization: ensure finite, deterministic output
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)
    
    # Ensure shape (N,) even for N=1
    return score.astype(float)
