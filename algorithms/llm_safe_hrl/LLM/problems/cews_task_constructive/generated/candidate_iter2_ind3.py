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
    'Improved priority rule combining deadline-criticality gating, risk-aware energy efficiency,\n     starvation-aware fairness, and robust multi-scale normalization.\n\n     Key improvements over v1:\n     - Replaces exp(-slack) with sigmoid-based soft urgency for smoother transition near slack=0\n     - Introduces *critical-path energy leverage*: upward_rank weighted by normalized energy_efficiency\n     - Adds *risk-conditional waiting boost*: only activates when slack <= 0 OR uncertainty > median_uncertainty\n     - Uses *symmetric IQR scaling* with explicit fallback to std (not mean-abs) for better variance alignment\n     - Normalizes min_incremental_energy *relative to workload* (energy_per_MI) for fair cross-task comparison\n     - All operations eps-protected, nan-to-num guarded, and strictly finite.'
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    
    # Robust normalization: IQR-based, fallback to std if IQR near zero
    def robust_normalize(x):
        q1, q3 = np.quantile(x, [0.25, 0.75], method='midpoint')
        iqr = q3 - q1
        if iqr < eps:
            std_val = np.std(x) + eps
            return (x - np.mean(x)) / std_val
        return (x - np.median(x)) / (iqr + eps)
    
    # Soft deadline urgency: sigmoid centered at slack=0, steepness tuned for sensitivity
    slack_urgency = 1.0 / (1.0 + np.exp(slack / (np.maximum(np.std(slack), eps) + eps)))
    
    # Energy efficiency per computational work (joules per MI), capped to avoid extremes
    work_safe = np.maximum(remaining_work, eps)
    energy_per_mi = min_incremental_energy / work_safe
    capped_energy_per_mi = np.clip(energy_per_mi, 1e-6, 1e6)
    norm_energy_per_mi = robust_normalize(capped_energy_per_mi)
    
    # Critical-path energy leverage: prioritize high-rank tasks that are also energy-efficient
    # Scaled by normalized energy_per_mi (lower is better → negative weight)
    leveraged_rank = upward_rank * (1.0 - 0.5 * norm_energy_per_mi)
    
    # Duration-aware cost: sqrt(exec + comm) penalizes long-latency tasks moderately
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    time_cost = np.sqrt(duration)
    norm_time_cost = robust_normalize(time_cost)
    
    # Uncertainty boost: only activate when slack is tight OR uncertainty is high
    median_uncert = np.median(uncertainty) if len(uncertainty) > 1 else 0.0
    risk_activation = np.logical_or(slack <= 0, uncertainty > (median_uncert + eps))
    norm_uncertainty = np.where(risk_activation, robust_normalize(uncertainty), 0.0)
    
    # Starvation guard: wait-time boost only under risk or high criticality
    median_rank = np.median(upward_rank) if len(upward_rank) > 1 else 0.0
    wait_activation = np.logical_or(slack <= 0, upward_rank > (median_rank + eps))
    norm_wait = np.where(wait_activation, robust_normalize(ready_wait_time), 0.0)
    
    # Composite score: smaller = higher priority
    # Weights emphasize urgency (negative), efficiency (negative), and fairness (negative)
    score = (
        +1.8 * slack_urgency              # Urgency: higher when slack low → lower score desired → negate later
        - 1.2 * robust_normalize(leveraged_rank)  # Critical efficiency: prefer high rank & low energy/Mi
        + 0.4 * norm_time_cost            # Mild penalty for long duration
        + 0.3 * norm_uncertainty          # Risk-aware boost for uncertain tasks under pressure
        - 0.2 * norm_wait                 # Fairness: reduce score for starved tasks (higher priority)
        + 0.1 * robust_normalize(min_exec_time)   # Small exec-time bias
        + 0.05 * robust_normalize(min_comm_time)  # Small comm-time bias
    )
    
    # Ensure final score is finite and deterministic
    return np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
