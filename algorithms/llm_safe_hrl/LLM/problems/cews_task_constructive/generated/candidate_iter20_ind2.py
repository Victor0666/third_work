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
    Self-evolved v2: Hybrid critical-path + deadline-aware energy fairness with starvation-robust normalization.
    
    Key innovations:
    - Combines Parent 2's outlier-robust division-based risk weighting and slack-aware energy scaling
    - Integrates Parent 1's hard-deadline clamp + soft urgency decay for unambiguous boundary behavior
    - Uses hybrid normalization: MAD for N>=3, quantile-based (Q1/Q3) robust scaling otherwise for better small-N stability
    - Introduces *critical-path density synergy*: multiplies cp_density by (1 + uncertainty) only when slack <= 0 to amplify risky paths
    - Adds *wait-aware energy fairness*: penalizes high-energy tasks proportionally to wait time under feasibility constraint
    - All operations protected against zero/Nan/inf; deterministic and bounded output
    """
    eps = 1e-08
    # Cast and sanitize inputs
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
    
    # Clip and sanitize all inputs to avoid NaN/inf/zero issues
    min_exec_time = np.clip(np.nan_to_num(min_exec_time, nan=eps, posinf=1e9, neginf=eps), eps, 1e9)
    min_comm_time = np.clip(np.nan_to_num(min_comm_time, nan=eps, posinf=1e9, neginf=eps), eps, 1e9)
    min_incremental_energy = np.clip(np.nan_to_num(min_incremental_energy, nan=eps, posinf=1e9, neginf=eps), eps, 1e9)
    slack = np.nan_to_num(slack, nan=0.0, posinf=1e9, neginf=-1e9)
    upward_rank = np.clip(np.nan_to_num(upward_rank, nan=eps, posinf=1e6, neginf=eps), eps, 1e6)
    remaining_work = np.clip(np.nan_to_num(remaining_work, nan=eps, posinf=1e9, neginf=eps), eps, 1e9)
    ready_wait_time = np.clip(np.nan_to_num(ready_wait_time, nan=0.0, posinf=1e6, neginf=0.0), 0.0, 1e6)
    uncertainty = np.clip(np.nan_to_num(uncertainty, nan=0.0, posinf=1e3, neginf=0.0), 0.0, 1e3)
    
    # Compute base metrics
    task_duration = min_exec_time + min_comm_time
    feasible_mask = slack >= -eps
    
    # Hard-deadline urgency: strict penalty floor at slack=0, exponential decay for slack>0
    urgency_hard = np.full(N, 2.0)
    urgency_soft = np.exp(-np.clip(slack / (task_duration + eps), 0.0, 20.0))
    deadline_urgency = np.where(slack <= 0, urgency_hard, np.clip(urgency_soft, 0.1, 1.0))
    
    # Risk-weighted critical alignment: Parent 2's robust division + Parent 1's slack amplification
    slack_risk_penalty = 1.0 + np.maximum(0.0, -slack) * uncertainty
    crit_alignment = upward_rank / (slack_risk_penalty * (min_incremental_energy + eps))
    crit_alignment = np.clip(crit_alignment, eps, 1e7)
    
    # Slack-aware energy fairness: Parent 2's scaling, enhanced with wait-aware penalty
    slack_factor = np.maximum(1.0, 1.0 + slack / 10.0)
    work_slack_scaled = remaining_work * slack_factor
    energy_per_work_scaled = min_incremental_energy / (work_slack_scaled + eps)
    energy_per_work_scaled = np.clip(energy_per_work_scaled, eps, 1e9)
    
    # Critical-path density: Parent 2's term, amplified under risk (Parent 1 style)
    cp_density = upward_rank / (task_duration + eps)
    cp_density = np.clip(cp_density, eps, 1e7)
    cp_density_amplified = np.where(slack <= 0, cp_density * (1.0 + uncertainty), cp_density)
    
    # Wait-aware energy penalty: only for feasible tasks, normalized by wait time
    wait_energy_penalty = np.zeros_like(min_incremental_energy)
    if np.any(feasible_mask):
        feasible_wait = ready_wait_time[feasible_mask]
        feasible_energy = min_incremental_energy[feasible_mask]
        # Normalize wait per feasible set to avoid outliers
        median_wait_f = np.median(feasible_wait) + eps
        wait_norm = np.where(feasible_mask, ready_wait_time / (median_wait_f + eps), 0.0)
        wait_energy_penalty = wait_norm * energy_per_work_scaled * 0.5
    
    # Robust normalization: quantile-based for small N, MAD for larger N
    def robust_normalize(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, -1e9, 1e9)
        if N == 1:
            return np.array([0.0])
        elif N < 3:
            # Quantile-based centering: more stable than min-max for tiny samples
            q1, q3 = np.quantile(x, [0.25, 0.75], method='midpoint')
            iqr = q3 - q1 + eps
            center = np.median(x)
            z = (x - center) / iqr
            return np.clip(z, -4.0, 4.0)
        else:
            # MAD for robustness
            center = np.median(x)
            abs_devs = np.abs(x - center)
            mad = np.median(abs_devs) + eps
            z = (x - center) / mad
            return np.clip(z, -6.0, 6.0)
    
    norm_urgency = robust_normalize(deadline_urgency)
    norm_crit = robust_normalize(crit_alignment)
    norm_energy = robust_normalize(energy_per_work_scaled)
    norm_cp_density = robust_normalize(cp_density_amplified)
    norm_wait_energy = robust_normalize(wait_energy_penalty)
    
    # Final score: prioritize deadline urgency and critical alignment, balance with energy and density
    # Lower score = higher priority
    score = (
        0.45 * norm_urgency 
        - 0.35 * norm_crit 
        + 0.12 * norm_energy 
        + 0.08 * norm_cp_density 
        + 0.05 * norm_wait_energy
    )
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
