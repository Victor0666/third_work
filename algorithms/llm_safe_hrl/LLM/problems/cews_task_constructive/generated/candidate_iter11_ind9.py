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
    v2 priority rule: Combines crisp urgency gating (P2) with robust wait-aware fairness (P1),
    adds criticality-normalized energy efficiency, and introduces slack-conditional uncertainty coupling.
    
    Key innovations:
      - Absolute urgency flag (slack <= 0 → 0.0 score) ensures hard DDL compliance
      - Energy term uses *criticality-weighted energy density* normalized by upward_rank, 
        then thresholded against workflow-relative median to avoid penalizing critical-efficiency tasks
      - Starvation guard merges P1's percentile-gated wait boost with P2's slack-aware decay:
        only activates when slack > -0.02 AND remaining_work > 20%-ile → prevents starvation without violating DDL
      - Uncertainty is coupled with both slack (via |slack|^{-1} clamped scaling) AND execution time 
        to prioritize uncertainty reduction where it most impacts risk completion time
      - All normalizations use outlier-robust 0.5%-99.5% clipping + min-max, with fallback for degenerate ranges
      - Final weights enforce strict hierarchy: urgency (0.48) > energy (0.20) > latency (0.13) > 
        fairness (0.09) > uncertainty modulation (0.06) > baseline uncertainty (0.04)
    """
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)
    
    def robust_minmax_norm(x):
        x_min = np.min(x)
        x_max = np.max(x)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        p005 = np.percentile(x, 0.5)
        p995 = np.percentile(x, 99.5)
        x_clipped = np.clip(x, p005, p995)
        x_min_c = np.min(x_clipped)
        x_max_c = np.max(x_clipped)
        if x_max_c - x_min_c < eps:
            return np.zeros_like(x)
        return (x_clipped - x_min_c) / (x_max_c - x_min_c + eps)
    
    # Crisp urgency: zero score for any task at or past deadline → highest priority
    urgency_flag = np.where(slack <= 0, 0.0, 1.0)
    
    # Duration and energy density
    duration = min_exec_time + min_comm_time + eps
    energy_density = min_incremental_energy / (duration + eps)
    
    # Criticality-normalized efficiency: energy_density per unit upward_rank
    # Thresholded against workflow-relative median to protect critical-path efficiency
    eff_per_rank = energy_density / (upward_rank + eps)
    threshold_eff_per_rank = np.median(eff_per_rank) + eps
    energy_penalty_mask = (eff_per_rank > threshold_eff_per_rank).astype(float)
    norm_energy_density = robust_minmax_norm(energy_density)
    energy_penalty = norm_energy_density * energy_penalty_mask * urgency_flag
    
    # Latency term: execution + communication, weighted by criticality and uncertainty
    comm_weight = 1.0 + 0.5 * robust_minmax_norm(upward_rank) + 0.3 * robust_minmax_norm(uncertainty)
    weighted_comm = min_comm_time * comm_weight
    latency_raw = min_exec_time + weighted_comm + eps
    norm_latency = robust_minmax_norm(latency_raw)
    
    # Fairness/starvation guard: only activates when safe (slack > -0.02) AND task is non-trivial
    slack_safe_mask = (slack > -0.02).astype(float)
    work_threshold = np.quantile(remaining_work, 0.2) + eps if N > 1 else np.median(remaining_work) + eps
    work_sufficient_mask = (remaining_work > work_threshold).astype(float)
    starvation_gate = slack_safe_mask * work_sufficient_mask
    wait_decay = np.exp(-0.5 * ready_wait_time / (np.abs(slack) + 1.0))
    norm_wait_time = robust_minmax_norm(ready_wait_time)
    wait_penalty = starvation_gate * norm_wait_time * wait_decay
    
    # Uncertainty coupling: scaled by both slack proximity and execution time impact
    slack_abs = np.abs(slack) + 1.0
    slack_scale_factor = np.clip(1.0 / slack_abs, 0.1, 10.0)
    # Prioritize uncertainty reduction where exec time dominates duration
    exec_ratio = min_exec_time / (duration + eps)
    uncertainty_boost = uncertainty * slack_scale_factor * (0.5 + 0.5 * exec_ratio)
    norm_uncertainty_boost = robust_minmax_norm(uncertainty_boost)
    
    # Baseline uncertainty penalty (low weight)
    norm_uncertainty = robust_minmax_norm(uncertainty)
    
    # Work-based fairness: higher remaining work → lower priority (to balance load)
    norm_remaining_work = robust_minmax_norm(remaining_work)
    
    # Final score: smaller = better; urgency dominates, others refine within feasible set
    score = (
        0.48 * urgency_flag +
        0.20 * energy_penalty +
        0.13 * norm_latency +
        0.09 * wait_penalty +
        0.06 * norm_uncertainty_boost +
        0.04 * norm_uncertainty +
        0.04 * norm_remaining_work
    )
    
    # Clip and sanitize
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    
    return score
