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
    Self-evolved priority rule v2: Deadline-robust critical-path leverage + adaptive risk-energy coupling + starvation-aware fairness.
    
    Key integrations & improvements:
    - Combines Parent 2's robust trimmed-mean centrality and slack-aware energy scaling with Parent 1's monotonic deadline penalty and critical-path leverage ratio.
    - Replaces sigmoid urgency with strictly monotonic linear urgency (Parent 1) for ordering stability, but retains Parent 2's trimmed-mean normalization for robustness.
    - Uses critical-path leverage = upward_rank / (remaining_work + eps) as primary CP importance metric (Parent 1), scaled by lateness pressure (Parent 2).
    - Introduces *risk-weighted energy efficiency*: min_incremental_energy / (task_duration * (1 + uncertainty)) — directly penalizes high-uncertainty high-energy tasks.
    - Fairness via *adaptive wait boost*: activated only when task is both non-urgent (slack >= 0) AND wait_per_work > median + 1.5*MAD (more robust than std).
    - All normalizations use bounded robust_minmax with trimming; all divisions guarded; NaN/inf handled deterministically.
    - Final score is convex combination of interpretable, monotonic, deadline-respecting components.
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
    
    def robust_minmax(x):
        if x.size == 0:
            return np.zeros_like(x)
        x_sorted = np.sort(x)
        trim_n = max(1, int(0.1 * len(x_sorted)))
        x_trimmed = x_sorted[trim_n:-trim_n] if len(x_sorted) > 2 * trim_n else x_sorted
        x_min, x_max = np.min(x_trimmed), np.max(x_trimmed)
        rng = x_max - x_min + eps
        return np.clip((x - x_min) / rng, 0.0, 1.0)
    
    def robust_mad(x):
        if x.size == 0:
            return eps
        med = np.median(x)
        abs_dev = np.abs(x - med)
        mad = np.median(abs_dev)
        return mad if mad > eps else np.mean(abs_dev) + eps
    
    # Core metrics
    task_duration = min_exec_time + min_comm_time + eps
    neg_slack = np.maximum(-slack, 0.0)
    uncertainty_factor = 1.0 + 0.8 * np.clip(uncertainty, 0.0, 3.0)
    deadline_penalty = neg_slack * uncertainty_factor  # monotonic, Parent 1 style
    
    # Critical-path leverage: impact per unit work (Parent 1), scaled by lateness pressure (Parent 2)
    cp_leverage = upward_rank / (remaining_work + eps)
    cp_leverage_pressure = cp_leverage * (1.0 + np.clip(neg_slack / (np.median(task_duration) + eps), 0.0, 3.0))
    
    # Risk-weighted energy efficiency: lower is better; penalized by duration & uncertainty
    energy_efficiency = min_incremental_energy / (task_duration * (1.0 + np.clip(uncertainty, 0.0, 2.0)) + eps)
    
    # Starvation fairness: only for non-urgent tasks, using MAD-based robust z-score
    wait_per_work = ready_wait_time / (remaining_work + eps)
    wait_median = np.median(wait_per_work)
    wait_mad = robust_mad(wait_per_work)
    wait_z_threshold = wait_median + 1.5 * wait_mad
    is_starvable = (slack >= 0.0) & (wait_per_work > wait_z_threshold)
    starvation_boost = np.where(is_starvable, ready_wait_time, 0.0)
    
    # Normalize components
    norm_deadline = robust_minmax(deadline_penalty)
    norm_cp = robust_minmax(cp_leverage_pressure)
    norm_energy = robust_minmax(energy_efficiency)
    norm_starvation = robust_minmax(starvation_boost)
    
    # Final convex combination: prioritize deadline compliance first, then CP impact, then energy, then fairness
    # Coefficients sum to 1.0 and reflect objective hierarchy: DDL hard constraint > critical path > energy > fairness
    score = (
        0.48 * norm_deadline + 
        0.22 * (1.0 - norm_cp) +  # higher cp_leverage_pressure → lower score (priority)
        0.18 * norm_energy + 
        0.12 * norm_starvation
    )
    
    # Ensure finite output and correct shape
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
