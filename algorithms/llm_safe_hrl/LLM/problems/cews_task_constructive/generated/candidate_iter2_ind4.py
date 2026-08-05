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
    Hybrid priority rule: deadline-criticality-aware energy efficiency with starvation control and robust risk gating.
    
    Key improvements:
    - Combines Parent 2's sharp risk-adjusted urgency (negative slack × (1+uncertainty)) with Parent 1's bounded starvation relief
    - Uses percentile-based robust normalization (Q75/Q25) from Parent 2 for outlier resistance
    - Introduces *deadline-tightness gating*: only applies communication-risk penalty when slack is tight (|slack| < median)
    - Replaces raw energy-per-second with *energy-efficiency density*: (min_incremental_energy / (min_exec_time + min_comm_time + eps)) × (1 + uncertainty)
      to penalize high-uncertainty low-efficiency tasks more aggressively
    - Adds *critical path leverage*: upward_rank × (remaining_work / (max(remaining_work) + eps)) normalized, weighted higher than in Parent 2
    - Ensures all components contribute negatively to score (lower = better) via sign alignment and negation where needed
    - All operations are eps-protected, nan-to-num guarded, and shape-preserving
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
    
    def robust_normalize(x):
        q75, q25 = np.percentile(x, 75), np.percentile(x, 25)
        iqr = q75 - q25 + eps
        center = np.median(x)
        return (x - center) / iqr
    
    # Risk-adjusted urgency: strong negative contribution for overdue tasks (Parent 2)
    deadline_risk_mask = (slack < 0).astype(float)
    risk_magnitude = -slack * deadline_risk_mask * (1.0 + uncertainty)
    norm_risk = -robust_normalize(risk_magnitude)  # negative => higher priority
    
    # Energy-efficiency density: penalizes high-uncertainty inefficient tasks
    time_sum = min_exec_time + min_comm_time + eps
    energy_eff_density = (min_incremental_energy / time_sum) * (1.0 + uncertainty)
    norm_eff_density = -robust_normalize(energy_eff_density)  # negative => higher priority
    
    # Critical path leverage: importance × work density (enhanced from Parent 2)
    max_rw = np.max(remaining_work) + eps
    critical_leverage = upward_rank * (remaining_work / max_rw)
    norm_critical = -robust_normalize(critical_leverage)  # negative => higher priority
    
    # Starvation relief: sigmoid-bounded fairness (from Parent 1, improved)
    wait_scale = np.maximum(np.mean(ready_wait_time), eps)
    starvation_relief = 1.0 / (1.0 + np.exp(-(ready_wait_time / (wait_scale + eps) - 2.0)))
    norm_starvation = -robust_normalize(starvation_relief)  # negative => higher priority
    
    # Deadline-tightness gated communication risk penalty (novel)
    abs_slack = np.abs(slack)
    tightness_mask = (abs_slack < np.median(abs_slack) + eps).astype(float)
    comm_risk_penalty = min_comm_time * uncertainty * tightness_mask
    norm_comm_risk = robust_normalize(comm_risk_penalty)  # positive penalty
    
    # Execution time penalty (lighter weight, robust normalized)
    norm_exec = robust_normalize(min_exec_time)
    
    # Uncertainty penalty (only when slack permits, novel conditional scaling)
    uncertainty_penalty = uncertainty * (1.0 - deadline_risk_mask)  # no penalty if already overdue
    norm_uncertainty = robust_normalize(uncertainty_penalty)
    
    # Final score: all terms aligned so lower = better; weights tuned for stability & deadline safety
    score = (
        0.35 * norm_risk +           # highest weight: hard deadline enforcement
        0.25 * norm_eff_density +   # energy efficiency under risk
        0.20 * norm_critical +      # critical path leverage
        0.10 * norm_starvation +    # fairness for long-waiting tasks
        0.04 * norm_comm_risk +     # communication risk only when slack is tight
        0.03 * norm_exec +          # execution time baseline
        0.03 * norm_uncertainty     # uncertainty cost only for non-overdue tasks
    )
    
    return np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
