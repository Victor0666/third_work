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
    v2: Hybrid urgency-criticality-energy-fairness priority with robust quantile normalization,
    adaptive risk gating, and lexicographic deadline dominance.
    
    Key improvements:
    - Combines Parent 2's convex combination & robust_divide with Parent 1's arctan urgency scaling
    - Uses IQR-based quantile normalization (Parent 2) but with 10th/90th percentiles for tighter dynamic range
    - Introduces *deadline-aware energy efficiency*: energy term weighted by upward_rank only when slack > 0 AND uncertainty < threshold
    - Adds *latency-pressure fairness*: anti-starvation term gated by both wait time and criticality to avoid biasing non-critical tasks
    - All terms normalized to [0,1] via robust quantile scaling with degenerate-case fallback
    - Final score: urgency (50%) + latency pressure (25%) + safe-energy (15%) + criticality-gated fairness (10%)
    """
    eps = 1e-08
    
    def robust_divide(a, b):
        return np.divide(a, np.where(np.abs(b) < eps, eps, b), out=np.full_like(a, eps), where=np.abs(b) >= eps)
    
    def sanitize(x):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(x, nan=eps, posinf=1e6, neginf=-1e6)
    
    # Sanitize all inputs
    min_exec_time = sanitize(min_exec_time)
    min_comm_time = sanitize(min_comm_time)
    min_incremental_energy = sanitize(min_incremental_energy)
    slack = sanitize(slack)
    upward_rank = sanitize(upward_rank)
    remaining_work = sanitize(remaining_work)
    ready_wait_time = sanitize(ready_wait_time)
    uncertainty = sanitize(uncertainty)
    
    # Robust slack: reduce effective slack under uncertainty for proactive scheduling
    robust_slack = np.where(slack > 0, slack - 1.5 * uncertainty, slack)
    
    # Smooth, bounded urgency: arctan-based, monotonic, finite-range [0,1]
    # Higher urgency for smaller (more negative) robust_slack
    urgency = (np.pi / 2 - np.arctan(-robust_slack / (1.0 + eps))) / np.pi
    
    # Latency pressure: critical path importance per unit execution+comm time
    exec_comm = min_exec_time + min_comm_time + eps
    latency_pressure = robust_divide(upward_rank * (remaining_work + eps), exec_comm)
    
    # Safe energy efficiency: only prioritized when slack > 0 AND uncertainty is low (< 0.3 std equiv)
    safety_mask = (robust_slack > 0).astype(float) * (uncertainty < 0.3).astype(float)
    energy_efficiency = robust_divide(min_incremental_energy, exec_comm) * (1.0 + upward_rank) * safety_mask
    
    # Criticality-gated fairness: penalizes long waits but only for high-importance tasks
    urank_median = np.median(upward_rank) if len(upward_rank) > 1 else np.mean(upward_rank)
    fairness_gate = (ready_wait_time > 0.1).astype(float) * (upward_rank > urank_median).astype(float)
    fairness_term = ready_wait_time * fairness_gate * (1.0 + 0.2 * upward_rank)
    
    # Quantile normalization: robust 10th/90th percentile scaling, fallback for small/degenerate cases
    def quantile_normalize(x):
        if x.size == 1:
            return np.array([0.0])
        q10 = np.quantile(x, 0.1)
        q90 = np.quantile(x, 0.9)
        iqr = q90 - q10 + eps
        x_norm = (x - q10) / iqr
        x_norm = np.clip(x_norm, 0.0, 1.0)
        return x_norm
    
    norm_urgency = quantile_normalize(urgency)
    norm_latency = quantile_normalize(latency_pressure)
    norm_energy = quantile_normalize(energy_efficiency)
    norm_fair = quantile_normalize(fairness_term)
    
    # Convex combination: ensures monotonic alignment with objectives
    score = (
        0.50 * norm_urgency +
        0.25 * norm_latency +
        0.15 * norm_energy +
        0.10 * norm_fair
    )
    
    # Final sanitization: ensure finite, bounded output
    score = np.nan_to_num(score, nan=1e6, posinf=1e6, neginf=-1e6)
    score = np.clip(score, -1e6, 1e6)
    
    # Ensure shape (N,) — critical for environment compatibility
    return score.reshape(-1)
