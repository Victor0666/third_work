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
    v4 priority rule: Hard-deadline lockstep + adaptive urgency bands + 
    quadratic risk-adjusted energy density + latency-aware starvation guard +
    critical-path slack sensitivity + workload-normalized uncertainty coupling +
    outlier-resilient MAD normalization + explicit lateness penalty enforcement.

    Key synthesis improvements:
    - Retains Parent 2's dynamic urgency bands (urgent/critical/relaxed) for graded deadline pressure.
    - Adopts Parent 1's explicit unclipped linear lateness penalty for strict hard-deadline enforcement.
    - Combines Parent 2's quadratic uncertainty penalty on energy density with Parent 1's robust inverse-slack magnitude.
    - Enhances fairness: wait_efficiency relief now gated by *both* relaxed band AND normalized slack distance to median,
      avoiding over-penalization when slack is tight but positive.
    - Introduces *critical-path-aware slack pressure*: uses upward_rank-weighted slack deficit (not raw slack) to prioritize
      high-criticality tasks under pressure, preserving structural importance.
    - Uses unified robust_zscore (MAD-based, clipped [-5,5]) for all normalized components — no fragile std fallbacks.
    - Adds uncertainty-weighted work consolidation: uncertainty scaled by (1 - norm_remaining_work) to favor stable VMs for heavy sub-DAGs.
    - Weights sum to 1.0: urgency (0.42) > energy-gated (0.23) > criticality (0.16) > fairness (0.09) > cp_slack (0.07) > unc_work (0.03).
    - Strict finite sanitization: all divisions guarded, NaN/inf replaced deterministically, no in-place mutation.
    """
    eps = 1e-08
    # Clean and copy inputs to avoid mutation
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    slack = np.nan_to_num(np.asarray(slack, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=np.float64)
    
    # Robust MAD-based z-score with clipping
    def robust_zscore(x):
        if x.size == 0:
            return np.zeros_like(x)
        med = np.median(x)
        dev = x - med
        mad = np.median(np.abs(dev))
        if mad < eps:
            return np.zeros_like(x)
        z = dev / (mad + eps)
        return np.clip(z, -5.0, 5.0)
    
    # Dynamic urgency bands
    median_slack = np.median(slack) if N > 0 else 0.0
    is_urgent = (slack <= 0.0).astype(np.float64)
    is_critical = ((slack > 0.0) & (slack <= median_slack)).astype(np.float64)
    is_relaxed = (slack > median_slack).astype(np.float64)
    
    # Explicit lateness penalty: linear, unclipped, dominates score when violated
    slack_deficit = np.maximum(-slack, 0.0)
    lateness_penalty = np.where(is_urgent, -1000000000000000.0 + 1000.0 * slack_deficit, 0.0)
    
    # Inverse-slack magnitude with uncertainty confidence scaling (Parent 1 style, improved)
    inv_slack_raw = np.divide(1.0, np.abs(slack) + eps, out=np.zeros_like(slack), where=np.abs(slack) + eps != 0)
    inv_slack_confidence = 1.0 / (1.0 + uncertainty + eps)
    inv_slack = inv_slack_raw * inv_slack_confidence
    inv_slack = np.clip(inv_slack, 0.01, 500.0)
    
    # Duration and risk-adjusted energy density (quadratic uncertainty penalty from Parent 2)
    duration = min_exec_time + min_comm_time + eps
    effective_duration = duration * (1.0 + uncertainty + eps) ** 2
    risk_adj_energy_density = np.divide(min_incremental_energy, effective_duration, 
                                         out=np.zeros_like(min_incremental_energy), 
                                         where=effective_duration != 0)
    risk_adj_energy_density = np.where(np.isfinite(risk_adj_energy_density), risk_adj_energy_density, 0.0)
    norm_energy = robust_zscore(risk_adj_energy_density)
    
    # Critical-path importance (normalized by max upward_rank)
    total_cp = np.max(upward_rank) if N > 0 else 1.0
    cp_importance = np.divide(upward_rank, total_cp + eps, out=np.zeros_like(upward_rank), where=total_cp + eps != 0)
    norm_cp_importance = robust_zscore(cp_importance)
    
    # Wait efficiency: only active in relaxed band and scaled by slack distance to median
    wait_efficiency = np.divide(ready_wait_time, remaining_work + eps, 
                                 out=np.zeros_like(ready_wait_time), 
                                 where=remaining_work + eps != 0)
    wait_efficiency = np.where(np.isfinite(wait_efficiency), wait_efficiency, 0.0)
    norm_wait_efficiency = robust_zscore(wait_efficiency)
    # Fairness mask: relaxed AND slack far from median → higher penalty
    norm_slack_dist = np.abs(slack - median_slack) / (np.std(slack) + eps) if N > 1 else np.zeros_like(slack)
    fairness_mask = is_relaxed * np.clip(norm_slack_dist, 0.0, 1.0)
    
    # Critical-path slack sensitivity: high-criticality tasks under deficit get priority
    cp_slack_sensitivity = cp_importance * slack_deficit
    norm_cp_slack_sensitivity = robust_zscore(cp_slack_sensitivity)
    
    # Workload-normalized uncertainty coupling: favor stable VMs for heavy work
    norm_remaining_work = robust_zscore(remaining_work)
    unc_work_density = uncertainty * (1.0 - np.clip(norm_remaining_work, 0.0, 1.0))
    unc_work_density = np.where(np.isfinite(unc_work_density), unc_work_density, 0.0)
    norm_unc_work_density = robust_zscore(unc_work_density)
    
    # Energy gate: apply energy penalty only in critical/relaxed bands (not urgent)
    energy_gate = is_critical + is_relaxed
    
    # Weighted combination (sums to 1.0)
    w_urgency = 0.42
    w_energy = 0.23
    w_critical = 0.16
    w_fairness = 0.09
    w_cp_slack = 0.07
    w_unc_work = 0.03
    
    base_score = (
        w_urgency * (-inv_slack) + 
        w_energy * (norm_energy * energy_gate) + 
        w_critical * (-norm_cp_importance) + 
        w_fairness * (norm_wait_efficiency * fairness_mask) + 
        w_cp_slack * (-norm_cp_slack_sensitivity) + 
        w_unc_work * norm_unc_work_density
    )
    
    # Apply lateness penalty unconditionally — overrides base score for urgent tasks
    score = np.where(is_urgent, lateness_penalty, base_score)
    
    # Final sanitization
    score = np.clip(score, -1000000000000000.0, 1000000000000000.0)
    score = np.nan_to_num(score, nan=1000000000000000.0, posinf=1000000000000000.0, neginf=-1000000000000000.0)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
