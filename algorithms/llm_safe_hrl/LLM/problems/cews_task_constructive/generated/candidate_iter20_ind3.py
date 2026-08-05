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
    v2 priority rule: Hard deadline lockstep + adaptive critical-path energy gating + 
    fairness-aware wait-efficiency + uncertainty-residual coupling + robust gradient urgency.
    
    Key innovations:
    - Strict DDL lockstep: slack <= 0 → score = -1e15 (guaranteed top priority)
    - Adaptive criticality gate: uses normalized slack margin relative to *critical path duration* (upward_rank-based estimate)
    - Energy penalty gated by both urgency AND criticality, scaled by energy-per-duration density
    - Wait-efficiency fairness via robust z-score of (ready_wait_time / (remaining_work + eps))
    - Uncertainty coupling: uncertainty * upward_rank / (|slack| + eps), clipped and normalized to suppress noise far from deadline
    - Gradient urgency: for non-urgent tasks, use smooth sigmoid over normalized slack deficit to preserve ordering near deadline boundary
    - All normalizations use percentile-clipped robust minmax (1%-99%) with fallbacks; no fragile median/MAD in latency-critical path
    - Final weights sum to 1.0: urgency (0.40) > energy-gated (0.25) > progress_velocity (0.15) > wait_efficiency (0.10) > uncertainty_slack (0.05) > criticality_bias (0.05)
    """
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=np.float64)
    min_comm_time = np.asarray(min_comm_time, dtype=np.float64)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slack = np.asarray(slack, dtype=np.float64)
    upward_rank = np.asarray(upward_rank, dtype=np.float64)
    remaining_work = np.asarray(remaining_work, dtype=np.float64)
    ready_wait_time = np.asarray(ready_wait_time, dtype=np.float64)
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    
    # Ensure finite values; replace NaN/inf with safe defaults
    inputs = [min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty]
    for arr in inputs:
        arr[:] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=np.float64)
    
    def robust_minmax_norm(x):
        if x.size == 0:
            return np.zeros_like(x)
        p01 = np.percentile(x, 1.0, method='midpoint')
        p99 = np.percentile(x, 99.0, method='midpoint')
        x_clipped = np.clip(x, p01, p99)
        x_min = np.min(x_clipped)
        x_max = np.max(x_clipped)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x_clipped - x_min) / (x_max - x_min + eps)
    
    # 1. Hard deadline lockstep: absolute priority for overdue or at-deadline tasks
    is_urgent = (slack <= 0.0).astype(np.float64)
    urgency_score = np.full(N, -1000000000000000.0, dtype=np.float64)
    
    # 2. Duration & critical path duration proxy
    duration = min_exec_time + min_comm_time + eps
    cp_duration = upward_rank + eps  # HEFT upward rank approximates critical path residual duration
    
    # 3. Gradient urgency for non-urgent tasks: smooth sigmoid over normalized slack deficit
    # Normalize slack deficit relative to typical duration to enable cross-workflow comparability
    norm_slack_deficit = np.divide(-slack, duration, out=np.zeros_like(slack), where=duration != 0)
    norm_slack_deficit = np.where(np.isfinite(norm_slack_deficit), norm_slack_deficit, 0.0)
    # Sigmoid centered at 0.5 (moderate urgency), steepness tuned for deadline proximity sensitivity
    gradient_urgency = 1.0 / (1.0 + np.exp(4.0 * (norm_slack_deficit - 0.5)))
    
    # 4. Critical-path energy gating: only penalize energy when slack is tight *relative to critical path*
    rel_slack_margin = np.divide(slack, cp_duration, out=np.zeros_like(slack), where=cp_duration != 0)
    rel_slack_margin = np.where(np.isfinite(rel_slack_margin), rel_slack_margin, 0.0)
    # Gate activates when slack is <= 50% of critical path duration (adaptive threshold)
    energy_gate = (rel_slack_margin <= 0.5).astype(np.float64)
    # Energy density: incremental energy per unit duration (efficiency-aware)
    energy_density = np.divide(min_incremental_energy, duration, out=np.zeros_like(min_incremental_energy), where=duration != 0)
    energy_density = np.where(np.isfinite(energy_density), energy_density, 0.0)
    norm_energy_density = robust_minmax_norm(energy_density)
    energy_penalty = norm_energy_density * energy_gate
    
    # 5. Progress velocity: upward_rank / duration → high-impact low-latency tasks prioritized
    progress_velocity = np.divide(upward_rank, duration, out=np.zeros_like(upward_rank), where=duration != 0)
    norm_progress_velocity = robust_minmax_norm(progress_velocity)
    
    # 6. Fairness: wait-efficiency = wait time per unit remaining work, robustly normalized
    wait_efficiency = np.divide(ready_wait_time, remaining_work + eps, out=np.zeros_like(ready_wait_time), where=remaining_work + eps != 0)
    wait_efficiency = np.where(np.isfinite(wait_efficiency), wait_efficiency, 0.0)
    norm_wait_efficiency = robust_minmax_norm(wait_efficiency)
    
    # 7. Uncertainty-residual coupling: uncertainty scaled by criticality and dampened by slack distance
    abs_slack = np.abs(slack) + eps
    unc_slack_ratio = np.divide(uncertainty, abs_slack, out=np.zeros_like(uncertainty), where=abs_slack != 0)
    unc_rank_scaled = unc_slack_ratio * upward_rank
    # Clip to avoid explosion near zero slack while preserving signal at moderate uncertainty
    unc_sensitivity = np.clip(unc_rank_scaled, 0.0, 5000.0)
    norm_unc_sensitivity = robust_minmax_norm(unc_sensitivity)
    
    # 8. Criticality bias: upward_rank normalized to reinforce structural importance uniformly
    norm_upward_rank = robust_minmax_norm(upward_rank)
    
    # Combine components with interpretable weights (sum = 1.0)
    # Non-urgent branch: gradient urgency dominates; others provide fine-grained tradeoff
    base_score = (
        0.40 * (1.0 - gradient_urgency) +           # smaller gradient_urgency → higher priority → lower score
        0.25 * energy_penalty +
        0.15 * (1.0 - norm_progress_velocity) +     # higher velocity → higher priority → lower score
        0.10 * norm_wait_efficiency +
        0.05 * norm_unc_sensitivity +
        0.05 * (1.0 - norm_upward_rank)             # higher upward_rank → higher priority → lower score
    )
    
    # Apply hard deadline lockstep: urgent tasks override all else
    score = np.where(is_urgent, urgency_score, base_score)
    
    # Final numeric safety
    score = np.clip(score, -1000000000000000.0, 1000000000000000.0)
    score = np.nan_to_num(score, nan=1000000000000000.0, posinf=1000000000000000.0, neginf=-1000000000000000.0)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
