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
    v2 priority rule: Hard urgency exclusivity + latency-gated criticality + 
                      risk-normalized energy + starvation-robust fairness + 
                      exponential slack sensitivity + local percentile robustness.
    
    Key synthesis:
    - Inherits Parent 2's strict urgency exclusivity and exponential slack decay for stability & deadline safety.
    - Adopts Parent 1's *local trimmed norm* (MAD-based) for small-N robustness, but with degenerate fallback to minmax when MAD=0.
    - Uses Parent 2's tighter critical-path gating (rel_slack <= 0.1) and work-aware fairness (wait-per-work).
    - Introduces novel *risk-adjusted energy density*: divides energy by effective duration (duration * (1 + uncertainty)), 
      normalized locally and gated only under tight slack AND high criticality.
    - Adds *deadline-proximity continuity*: smooth transition near slack=0 via proximity_bias, avoiding discontinuity.
    - All operations guarded against NaN/inf/zero; deterministic and shape-preserving.
    """
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=np.float64).copy()
    min_comm_time = np.asarray(min_comm_time, dtype=np.float64).copy()
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=np.float64).copy()
    slack = np.asarray(slack, dtype=np.float64).copy()
    upward_rank = np.asarray(upward_rank, dtype=np.float64).copy()
    remaining_work = np.asarray(remaining_work, dtype=np.float64).copy()
    ready_wait_time = np.asarray(ready_wait_time, dtype=np.float64).copy()
    uncertainty = np.asarray(uncertainty, dtype=np.float64).copy()
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=np.float64)

    def local_robust_norm(x):
        x_clean = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if N == 1:
            return np.zeros_like(x_clean)
        x_med = np.median(x_clean)
        abs_dev = np.abs(x_clean - x_med)
        mad = np.median(abs_dev)
        if mad < eps:
            # Fallback to minmax for degenerate case
            x_min = np.min(x_clean)
            x_max = np.max(x_clean)
            if x_max - x_min < eps:
                return np.zeros_like(x_clean)
            return (x_clean - x_min) / (x_max - x_min + eps)
        scale = 3.0 * mad + eps
        z = (x_clean - x_med) / scale
        return np.clip(z, -3.0, 3.0)

    is_urgent = (slack <= 0.0).astype(np.float64)
    tau = 10.0
    proximity_bias = np.exp(-np.maximum(0.0, slack) / tau)  # smooth penalty near deadline

    duration = min_exec_time + min_comm_time + eps
    rel_slack = np.divide(slack, duration, out=np.zeros_like(slack), where=duration != 0)
    rel_slack = np.nan_to_num(rel_slack, nan=0.0, posinf=0.0, neginf=0.0)

    # Critical latency: duration * upward_rank, robustly normalized
    critical_latency_raw = duration * upward_rank
    norm_critical_latency = local_robust_norm(critical_latency_raw)

    # Risk-adjusted energy density: energy / (duration * (1 + uncertainty))
    effective_duration = duration * (1.0 + uncertainty + eps)
    risk_energy_density = np.divide(min_incremental_energy, effective_duration, 
                                   out=np.zeros_like(min_incremental_energy), 
                                   where=effective_duration != 0)
    risk_energy_density = np.nan_to_num(risk_energy_density, nan=0.0, posinf=0.0, neginf=0.0)
    norm_energy_density = local_robust_norm(risk_energy_density)

    # Tight slack gating: stricter threshold (<= 0.1) AND high upward rank (top 75%)
    rank_threshold = np.percentile(upward_rank, 75.0) + eps if N > 1 else np.max(upward_rank) + eps
    tight_slack_mask = (rel_slack <= 0.1).astype(np.float64)
    high_rank_mask = (upward_rank > rank_threshold).astype(np.float64)
    energy_penalty_mask = tight_slack_mask * high_rank_mask

    # Fairness: normalized wait-per-work, penalizing starvation only above 10th percentile
    wait_per_work = np.divide(ready_wait_time, remaining_work + eps, 
                             out=np.zeros_like(ready_wait_time), 
                             where=remaining_work + eps != 0)
    wait_per_work = np.nan_to_num(wait_per_work, nan=0.0, posinf=0.0, neginf=0.0)
    wpw_finite = wait_per_work[np.isfinite(wait_per_work)]
    work_threshold = np.percentile(wpw_finite, 10.0) + eps if len(wpw_finite) > 0 else eps
    wait_gate = (wait_per_work >= work_threshold).astype(np.float64)
    norm_wait_per_work = local_robust_norm(wait_per_work)
    wait_penalty = (1.0 - is_urgent) * norm_wait_per_work * wait_gate

    # Uncertainty boost scaled by proximity_bias — amplifies risk awareness near deadline
    uncertainty_boost = uncertainty * proximity_bias
    norm_uncertainty_boost = local_robust_norm(uncertainty_boost)

    # Remaining work penalty: smaller work gets slight priority *only when non-urgent*
    norm_remaining_work = local_robust_norm(remaining_work)
    work_penalty = (1.0 - is_urgent) * (-norm_remaining_work)  # negative → favors small work

    # Weights tuned for dominance hierarchy: urgency > criticality > energy > fairness > uncertainty > work
    w_urgency = 0.45
    w_critical = 0.20
    w_energy = 0.15
    w_fairness = 0.10
    w_uncertainty = 0.07
    w_work = 0.03

    # Base score: start neutral, then apply urgency override
    score = np.full(N, 0.0, dtype=np.float64)
    score = np.where(is_urgent, -1e12, score)

    # Non-urgent contributions
    score = np.where(is_urgent, score, 
                     score + w_critical * norm_critical_latency +
                           w_energy * norm_energy_density * energy_penalty_mask +
                           w_fairness * wait_penalty +
                           w_uncertainty * norm_uncertainty_boost +
                           w_work * work_penalty)

    # Smooth deadline-proximity continuity: blend in proximity_bias as slack approaches zero
    score = np.where(is_urgent, score, 
                     score * (1.0 - proximity_bias) + (-1e12) * proximity_bias)

    # Final sanitization
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
