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
    'Improved priority rule combining deadline-risk gating, critical-path energy efficiency,\n    robust starvation mitigation, and uncertainty-aware criticality scaling.\n    \n    Key improvements:\n    - Uses *adaptive deadline-risk gating*: penalizes only slack < 0, but with exponential\n      amplification near deadline (e^-slack/tau) for sharper urgency transition.\n    - Combines Parent 2\'s IQR-based robust normalization with Parent 1\'s energy-per-work\n      ratio for better energy-efficiency signal under workload heterogeneity.\n    - Introduces *uncertainty-modulated criticality*: multiplies upward_rank by\n      (1 + uncertainty) only when slack < 0; otherwise dampens by (1 / (1 + uncertainty)).\n    - Replaces tanh wait boost with clipped sigmoid to ensure monotonic, bounded,\n      and interpretable starvation relief (0–1 range).\n    - All operations epsilon-protected, finite, deterministic, and shape-preserving.'
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    
    # Adaptive deadline risk: exponential urgency for negative slack, zero otherwise
    tau = np.clip(np.mean(np.abs(slack)) + eps, eps, 1000.0)
    deadline_risk = np.where(slack < 0, np.exp(-slack / tau), 0.0)
    
    # Uncertainty-modulated criticality: boost rank under risk, dampen when safe
    risk_factor = np.where(slack < 0, 1.0 + uncertainty, 1.0 / (1.0 + uncertainty + eps))
    critical_urgency = upward_rank * risk_factor * (remaining_work + eps)
    
    # Energy efficiency: marginal energy per unit of total time (exec + comm)
    total_time = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_second = min_incremental_energy / total_time
    
    # Starvation mitigation: soft, bounded, monotonic wait boost (0 to 1)
    median_wait = np.median(ready_wait_time) + eps
    wait_ratio = np.clip(ready_wait_time / (3.0 * median_wait + eps), 0.0, 10.0)
    wait_boost = 1.0 - 1.0 / (1.0 + np.exp(wait_ratio - 2.0))
    
    # Robust IQR-based normalization (Parent 2 style, simplified and stable)
    def robust_iqr_normalize(x):
        q75, q25 = np.percentile(x, [75, 25])
        iqr = q75 - q25
        scale = iqr if iqr > eps else np.mean(np.abs(x)) + eps
        return x / (scale + eps)
    
    norm_deadline_risk = robust_iqr_normalize(deadline_risk)
    norm_critical_urgency = robust_iqr_normalize(critical_urgency)
    norm_energy_eff = robust_iqr_normalize(energy_per_second)
    norm_wait_boost = robust_iqr_normalize(wait_boost)
    norm_exec = robust_iqr_normalize(min_exec_time)
    norm_comm = robust_iqr_normalize(min_comm_time)
    norm_uncert = robust_iqr_normalize(uncertainty)
    
    # Weighted score: prioritize deadline safety > critical path > energy > starvation
    score = (
        4.0 * norm_deadline_risk           # Strongest weight on urgent deadlines
        - 1.5 * norm_critical_urgency     # Favor high-impact, high-work tasks
        + 0.9 * norm_energy_eff           # Prefer low energy-per-second when safe
        + 0.7 * norm_wait_boost           # Moderate starvation relief
        + 0.15 * norm_exec                # Small penalty for long exec time
        + 0.1 * norm_comm                 # Small penalty for long comm time
        + 0.05 * norm_uncert              # Minimal direct uncertainty penalty
    )
    
    return np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
