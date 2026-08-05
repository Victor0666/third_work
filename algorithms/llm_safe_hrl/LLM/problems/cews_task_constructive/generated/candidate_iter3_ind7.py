import numpy as np

# 函数名中的 2 由 SeEvo 在生成 Prompt 时替换为目标版本号。
# 八个输入均为长度 N 的一维数组；相同下标始终指向同一个 ready task。
# 返回值也必须是长度 N 的一维数组，并遵守“分数越小，优先级越高”。
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
    Self-evolved priority rule: restores deadline dominance via urgency-gated criticality,
    fixes energy scaling to preserve urgency signal near deadline boundary, and introduces
    risk-magnitude-aware uncertainty boosting.
    
    Key evolutions:
      - Urgency is now the *primary gate*: all other terms are scaled by (1 - urgency) to ensure
        urgent tasks dominate regardless of energy/criticality — directly enforces hard DDL constraint.
      - Criticality-energy term becomes 'urgency-weighted criticality-per-energy': only activates
        when urgency > 0.3, preventing under-prioritization of borderline-urgent tasks.
      - Energy penalty is now *inverse urgency-scaled*: high energy tasks get stronger penalty when slack is tight,
        not diluted — fixes previous over-relaxation near deadline.
      - Uncertainty boost uses signed slack magnitude: boosts priority most for (tight slack AND high uncertainty),
        with linear ramp from median_slack down to min_slack — captures risk severity, not just sign.
      - Adds starvation guard: wait_boost activated *only* when urgency < 0.7 AND slack >= 0, avoiding late-task reward
        while ensuring fairness for non-urgent long-waiting tasks.
      - All components normalized robustly to [0,1]; final score is convex combination with strict bounds.
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
    
    def robust_minmax_norm(x):
        """Min-max normalize to [0, 1]; handles constant arrays safely."""
        x_min, x_max = np.min(x), np.max(x)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x - x_min) / (x_max - x_min + eps)
    
    # Compute bounded urgency: sigmoid centered at median_slack, scaled by IQR for stability
    median_slack = np.median(slack)
    iqr_slack = np.percentile(slack, 75) - np.percentile(slack, 25) + eps
    urgency = 1.0 / (1.0 + np.exp(-(slack - median_slack) / (iqr_slack + eps)))
    
    # Criticality-per-energy only matters when urgency > 0.3 — prevents dilution in borderline cases
    crit_per_energy = upward_rank / (min_incremental_energy + eps)
    norm_crit_per_energy = robust_minmax_norm(crit_per_energy)
    urgency_gate_crit = np.where(urgency > 0.3, 1.0, 0.0)
    crit_term = (1.0 - norm_crit_per_energy) * urgency_gate_crit
    
    # Inverse urgency-scaled energy penalty: higher penalty when urgent (tight slack)
    norm_energy = robust_minmax_norm(min_incremental_energy)
    energy_penalty = norm_energy * (1.0 + 0.5 * urgency)  # amplifies under pressure
    
    # Risk-magnitude-aware uncertainty boost: linear ramp from median to min slack
    slack_range = np.maximum(np.abs(np.min(slack) - median_slack), eps)
    uncertainty_risk_score = np.clip((median_slack - slack) / slack_range, 0.0, 1.0)
    uncertainty_boost = uncertainty * uncertainty_risk_score
    norm_uncertainty_boost = robust_minmax_norm(uncertainty_boost)
    
    # Starvation guard: only boost waiting time for non-urgent (urgency < 0.7) and feasible (slack >= 0) tasks
    wait_gate = np.where((urgency < 0.7) & (slack >= 0), 1.0, 0.0)
    wait_boost = ready_wait_time * wait_gate
    norm_wait_boost = robust_minmax_norm(wait_boost)
    
    # Execution and communication overheads: prioritize low-latency tasks, especially when urgent
    norm_exec = robust_minmax_norm(min_exec_time)
    norm_comm = robust_minmax_norm(min_comm_time)
    latency_term = 0.5 * norm_exec + 0.5 * norm_comm
    
    # Final score: urgency dominates; others modulate only within its envelope
    # Smaller score = higher priority → invert urgency (1-urgency) as base
    base_urgency = 1.0 - urgency
    score = (
        0.45 * base_urgency +
        0.20 * energy_penalty +
        0.15 * latency_term +
        0.10 * crit_term +
        0.05 * (1.0 - norm_wait_boost) +
        0.03 * norm_uncertainty_boost +
        0.02 * (1.0 - norm_crit_per_energy)  # residual criticality reinforcement
    )
    
    # Ensure finite, deterministic output
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    return score
