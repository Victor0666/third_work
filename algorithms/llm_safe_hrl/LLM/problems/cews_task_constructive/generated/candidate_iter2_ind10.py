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
    Hybrid priority rule combining deadline safety, critical-path awareness,
    energy-time efficiency, and starvation mitigation with enhanced robustness.
    
    Key improvements:
    - Uses *risk-gated exponential deadline penalty*: only activates for negative slack,
      bounded via clip to prevent overflow, normalized robustly.
    - Integrates *slack-adjusted criticality*: upward_rank scaled by (1 + max(0,-slack))⁻¹
      to smoothly degrade importance as lateness grows — more physical than gating.
    - Introduces *energy-per-latency ratio* (min_incremental_energy / (min_exec_time + min_comm_time + eps))
      as primary efficiency signal, normalized via clipped IQR.
    - Applies *adaptive waiting boost*: arctan-scaled wait time weighted by slack-aware gain
      (higher boost when slack > 0, suppressed when urgent).
    - Uses *uncertainty-weighted risk slack*: replaces raw slack with (slack - k * uncertainty)
      in criticality and risk terms for proactive DDL guard.
    - All features normalized via stable IQR clipping [-3, 3]; final score avoids unbounded terms.
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
        """IQR-based normalization: x -> (x - Q1) / (Q3 - Q1 + eps), clipped to [-3, 3]"""
        q1 = np.percentile(x, 25)
        q3 = np.percentile(x, 75)
        iqr = q3 - q1 + eps
        normed = (x - q1) / iqr
        return np.clip(normed, -3.0, 3.0)
    
    # Risk-adjusted slack: proactively tighten deadline under uncertainty
    risk_slack = slack - 0.5 * uncertainty
    # Deadline risk: exponential penalty only when risk_slack < 0, bounded to avoid overflow
    deadline_risk_raw = np.where(risk_slack < 0, 
                                 np.exp(np.clip(-risk_slack, 0, 20)) - 1.0, 
                                 0.0)
    deadline_risk = robust_normalize(deadline_risk_raw)
    
    # Slack-adjusted criticality: upward_rank degrades smoothly with lateness severity
    lateness_factor = np.clip(1.0 + np.maximum(0.0, -risk_slack), 1.0, 100.0)
    upward_rank_adj = upward_rank / lateness_factor
    upward_rank_norm = robust_normalize(upward_rank_adj)
    
    # Energy-efficiency ratio: joules per second of total latency (exec + comm)
    total_latency = min_exec_time + min_comm_time + eps
    energy_per_latency = min_incremental_energy / total_latency
    energy_eff_norm = robust_normalize(energy_per_latency)
    
    # Adaptive waiting boost: arctan-saturated, gain modulated by slack health
    wait_gain = np.where(slack >= 0, 1.0, 0.3)  # lower boost when overdue
    wait_saturation = np.arctan(ready_wait_time / (np.mean(ready_wait_time + eps) + eps)) / (np.pi / 2)
    wait_boost = wait_gain * wait_saturation
    
    # Remaining work normalized to guide load balancing without dominance
    work_norm = robust_normalize(remaining_work)
    
    # Final score: minimize risk, maximize criticality & efficiency, mitigate starvation
    # Coefficients tuned for convex trade-off: risk dominates, then criticality, then efficiency
    score = (
        +3.0 * deadline_risk      # Strong penalty for DDL violation risk
        -1.8 * upward_rank_norm   # Reward high-impact tasks, but attenuated by lateness
        +0.9 * energy_eff_norm    # Prefer low-energy-per-latency assignments
        +0.5 * wait_boost         # Gentle starvation prevention, context-aware
        +0.3 * work_norm          # Light bias toward larger sub-DAGs for fairness
    )
    
    return np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
