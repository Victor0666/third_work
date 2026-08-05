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
    v2: Refined deadline-dominant priority with adaptive fairness, resilient energy gating,
        and smoothed quantile normalization for small-N robustness.
    
    Key improvements over v1:
    - Replaces strict (wait ∧ criticality) fairness gate with *latency-aware fairness*: 
      fairness_term = ready_wait_time * sigmoid(upward_rank - median_rank), enabling smooth
      boost for moderately critical tasks and avoiding starvation of low-rank but urgent ones.
    - Relaxes energy safety mask to (robust_slack > -0.1) — allows energy-efficient assignment
      even under slight lateness risk if it prevents larger violations downstream.
    - Uses 5th/95th quantiles with fallback to min-max when IQR ≈ 0, improving stability for N < 4.
    - Introduces *urgency-smoothed energy coupling*: energy term scaled by urgency^(0.5) to 
      prioritize energy savings more strongly only when deadline pressure is moderate (not extreme).
    - Removes hard uncertainty threshold; instead uses continuous risk attenuation: 
      fairness_factor = exp(-uncertainty/2), energy_weight = exp(-uncertainty/3).
    - Final convex weights rebalanced: urgency (55%) > latency (20%) > safe-energy (15%) > fairness (10%)
      to reinforce lexicographic deadline dominance while preserving fairness & efficiency.
    """
    eps = 1e-08
    def robust_divide(a, b):
        return np.divide(a, np.where(np.abs(b) < eps, eps, b), out=np.full_like(a, eps), where=np.abs(b) >= eps)
    def sanitize(x):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(x, nan=eps, posinf=1000000.0, neginf=-1000000.0)
    
    min_exec_time = sanitize(min_exec_time)
    min_comm_time = sanitize(min_comm_time)
    min_incremental_energy = sanitize(min_incremental_energy)
    slack = sanitize(slack)
    upward_rank = sanitize(upward_rank)
    remaining_work = sanitize(remaining_work)
    ready_wait_time = sanitize(ready_wait_time)
    uncertainty = sanitize(uncertainty)
    
    # Robust slack: shrink positive slack by uncertainty, preserve negative slack as-is
    robust_slack = np.where(slack > 0, slack - 1.5 * uncertainty, slack)
    
    # Smooth bounded urgency: arctan-based, range [0,1], monotonic, finite
    urgency = (np.pi / 2 - np.arctan(-robust_slack / (1.0 + eps))) / np.pi
    
    # Execution+communication baseline (avoid division by zero)
    exec_comm = min_exec_time + min_comm_time + eps
    
    # Latency pressure: critical path work per unit time — higher = more urgent to schedule early
    latency_pressure = robust_divide(upward_rank * (remaining_work + eps), exec_comm)
    
    # Adaptive safety mask: relaxed to allow energy optimization even under mild lateness risk
    safety_mask = (robust_slack > -0.1).astype(float)
    
    # Risk-attenuated energy weight: smoother than hard threshold, decays exponentially with uncertainty
    energy_weight = np.exp(-uncertainty / 3.0)
    
    # Urgency-coupled energy efficiency: prioritize energy savings more when urgency is moderate
    # (urgency^0.5 suppresses energy term under extreme urgency, avoids delaying critical tasks for marginal savings)
    energy_efficiency = robust_divide(min_incremental_energy, exec_comm) * (1.0 + upward_rank) * safety_mask * (urgency ** 0.5) * energy_weight
    
    # Latency-aware fairness: smooth sigmoid gating centered at median upward_rank
    urank_med = np.median(upward_rank) if len(upward_rank) > 1 else np.mean(upward_rank)
    fairness_gate = 1.0 / (1.0 + np.exp(-(upward_rank - urank_med) / (np.std(upward_rank) + eps)))
    # Risk-attenuated fairness factor: reduce boost under high uncertainty
    fairness_factor = np.exp(-uncertainty / 2.0)
    fairness_term = ready_wait_time * fairness_gate * fairness_factor * (1.0 + 0.1 * upward_rank)
    
    def quantile_normalize(x):
        if x.size == 1:
            return np.array([0.0])
        q05 = np.quantile(x, 0.05)
        q95 = np.quantile(x, 0.95)
        iqr = q95 - q05 + eps
        # Fallback to min-max if dispersion is negligible (avoids division by near-zero)
        if iqr < eps * 100:
            x_min, x_max = np.min(x), np.max(x)
            x_norm = (x - x_min) / (x_max - x_min + eps)
        else:
            x_norm = (x - q05) / iqr
        x_norm = np.clip(x_norm, 0.0, 1.0)
        return x_norm
    
    norm_urgency = quantile_normalize(urgency)
    norm_latency = quantile_normalize(latency_pressure)
    norm_energy = quantile_normalize(energy_efficiency)
    norm_fair = quantile_normalize(fairness_term)
    
    # Weighted combination: reinforced deadline dominance
    score = 0.55 * norm_urgency + 0.20 * norm_latency + 0.15 * norm_energy + 0.10 * norm_fair
    
    # Final sanitization: ensure finite, bounded output
    score = np.nan_to_num(score, nan=1000000.0, posinf=1000000.0, neginf=-1000000.0)
    score = np.clip(score, -1000000.0, 1000000.0)
    
    return score.reshape(-1)
