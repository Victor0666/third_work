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
    Evolved v2: Deadline-hardened urgency + criticality-energy leverage + starvation-resilient fairness.
    
    Key synthesis:
    - Uses Parent 2's monotonic linear-soft urgency (stable, deadline-dominant)
    - Adopts Parent 1's slack-aware energy normalization *only for feasible tasks* to avoid distortion
    - Integrates Parent 2's direct starvation penalty but gates it only on feasibility and non-zero work
    - Introduces uncertainty-gated communication pressure (from Parent 1) but applies it *only* to urgent tasks (slack <= 0)
    - Combines criticality-energy tradeoff via risk-weighted efficiency ratio (Parent 2) with robust per-subset normalization
    - All normalizations use IQR with fallback to MAD for degenerate cases
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
    
    # Compute base metrics
    task_min_duration = np.maximum(min_exec_time + min_comm_time, eps)
    median_dur = np.median(task_min_duration) + eps
    median_ur = np.median(upward_rank) + eps
    median_work = np.median(remaining_work) + eps
    median_wait = np.median(ready_wait_time) + eps
    median_unc = np.median(uncertainty) + eps
    
    # === Urgency: monotonic linear-soft ramp (Parent 2) ===
    urgency_neg = 1.0 + np.clip(-slack / median_dur, 0.0, 1.0)
    urgency_pos = np.exp(-np.clip(slack / (median_dur + eps), 0.0, 20.0))
    deadline_urgency = np.where(slack <= 0, urgency_neg, urgency_pos)
    
    # === Criticality-Energy Efficiency Ratio ===
    # Risk-weighted energy: higher uncertainty → steeper penalty
    base_risk = np.maximum(0.0, -slack) / median_dur
    risk_exponent = np.clip(1.0 + 0.5 * base_risk + 0.3 * uncertainty, 1.0, 3.5)
    energy_risk_weighted = min_incremental_energy * np.power(1.0 + uncertainty, risk_exponent)
    energy_safe = np.maximum(energy_risk_weighted, eps)
    crit_eff_ratio = upward_rank / energy_safe
    crit_eff_ratio = np.clip(crit_eff_ratio, 1e-06, 1e6)
    
    # === Robust IQR normalization with MAD fallback ===
    def robust_iqr_norm(x):
        q1, q3 = np.quantile(x, [0.25, 0.75], method='midpoint')
        iqr = q3 - q1 + eps
        center = np.median(x)
        normed = (x - center) / iqr
        # Fallback to MAD if IQR is zero (degenerate case)
        if iqr < eps:
            mad = np.median(np.abs(x - center)) + eps
            normed = (x - center) / mad
        return np.clip(normed, -4.0, 4.0)
    
    norm_crit_eff = robust_iqr_norm(crit_eff_ratio)
    
    # === Slack-aware energy normalization (Parent 1 style) ===
    feasible_mask = slack >= 0.0
    energy_score = np.zeros_like(min_incremental_energy)
    if np.any(feasible_mask):
        feasible_energy = min_incremental_energy[feasible_mask]
        energy_center = np.median(feasible_energy)
        energy_mad = np.median(np.abs(feasible_energy - energy_center)) + eps
        energy_norm_base = (min_incremental_energy - energy_center) / energy_mad
        # Only reward low energy among feasible tasks; no penalty for overdue
        energy_score = np.where(feasible_mask, -np.clip(energy_norm_base, -4.0, 4.0), 0.0)
    else:
        energy_score = np.zeros(N)
    
    # === Uncertainty-gated communication pressure (Parent 1) applied only to urgent tasks ===
    comm_to_work_ratio = min_comm_time / (remaining_work + eps)
    urgent_high_uncert_mask = (slack <= 0.0) & (uncertainty > median_unc)
    comm_pressure = np.where(urgent_high_uncert_mask, comm_to_work_ratio * 3.0, comm_to_work_ratio * 0.1)
    comm_norm = robust_iqr_norm(comm_pressure)
    
    # === Starvation penalty: direct, normalized, feasibility-gated (Parent 2) ===
    starvation_mask = feasible_mask & (remaining_work > eps)
    wait_per_work = ready_wait_time / (remaining_work + eps)
    max_wait_pw = np.maximum(np.max(wait_per_work), eps)
    starvation_penalty = np.where(
        starvation_mask,
        np.clip(wait_per_work / max_wait_pw, 0.0, 1.0) * 0.4,
        0.0
    )
    
    # === Additional robust features ===
    norm_dur = robust_iqr_norm(task_min_duration)
    norm_work = robust_iqr_norm(remaining_work)
    norm_unc = robust_iqr_norm(uncertainty)
    
    # Final score: lower = better
    # Weights sum to 1.0; deadline urgency dominates, criticality-efficiency second, others fine-tune
    score = (
        0.45 * deadline_urgency +
        0.25 * (1.0 - norm_crit_eff) +
        0.10 * norm_dur +
        0.08 * energy_score +
        0.05 * comm_norm +
        0.04 * norm_work +
        0.02 * norm_unc +
        0.01 * starvation_penalty
    )
    
    # Final safeguard: finite, bounded, deterministic
    score = np.nan_to_num(score, nan=1e6, posinf=1e6, neginf=-1e6)
    score = np.clip(score, -1e6, 1e6)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
