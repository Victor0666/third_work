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
    Self-evolved priority rule: strict DDL-gating + adaptive urgency scaling + 
    starvation-aware wait boost with dynamic thresholding + energy dominance only under safety.
    Key improvements over v1:
    - Replaces fixed sigmoid scaling with *adaptive urgency gain*: slope of risk-sigmoid dynamically
      tuned by median_abs_slack to sharpen discrimination near critical deadlines without saturating.
    - Introduces *dynamic starvation threshold*: wait_boost activates only when ready_wait_time > 25%-ile,
      avoiding premature boosting of fresh tasks; uses sqrt-scaled boost for smoother, more stable growth.
    - Adds *latency-criticality coupling*: multiplies upward_rank_norm by (1 + total_latency_norm) to prioritize
      high-importance tasks that also contribute significantly to end-to-end latency — crucial for deadline adherence.
    - Uses *robust energy efficiency ratio*: replaces energy_per_latency with min_incremental_energy / (total_latency + eps)^0.8,
      reducing sensitivity to ultra-short tasks while preserving discriminative power.
    - All normalizations use IQR clipping [-3, 3]; final score is bounded, deterministic, and numerically safe.
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
        """IQR-based normalization clipped to [-3, 3] for stability and outlier resilience"""
        q1 = np.percentile(x, 25)
        q3 = np.percentile(x, 75)
        iqr = q3 - q1 + eps
        normed = (x - q1) / iqr
        return np.clip(normed, -3.0, 3.0)
    
    # Uncertainty-aware effective slack: preserves sign & relative ordering, damps noise
    effective_slack = slack / (1.0 + np.clip(uncertainty / 10.0, 0.0, 1000.0))
    
    # Adaptive urgency: sigmoid slope scales inversely with median |slack| → sharper near deadline
    median_abs_slack = np.abs(np.median(effective_slack)) + eps
    # Use steeper slope when slack is small (tight deadlines), shallower when large (loose)
    urgency_scale = np.clip(5.0 / (median_abs_slack + 0.1), 0.5, 10.0)
    risk_sigmoid = 1.0 / (1.0 + np.exp(-urgency_scale * effective_slack))
    deadline_risk_raw = np.where(effective_slack < 0, risk_sigmoid, 0.0)
    deadline_risk = robust_normalize(deadline_risk_raw)
    
    # Gated upward rank: full weight if slack >= 0; linear penalty ramp for negative slack (no explosion)
    upward_rank_gated = np.where(
        effective_slack >= 0,
        upward_rank,
        upward_rank * (1.0 + np.clip(-effective_slack, 0.0, 10.0))
    )
    
    # Latency-criticality coupling: amplify importance of high-rank tasks on long-latency paths
    total_latency = min_exec_time + min_comm_time + eps
    total_latency_norm = robust_normalize(total_latency)
    upward_rank_coupled = upward_rank_gated * (1.0 + 0.5 * total_latency_norm)
    upward_rank_norm = robust_normalize(upward_rank_coupled)
    
    # Robust energy efficiency: sublinear denominator avoids domination by ultra-short tasks
    energy_eff_ratio = min_incremental_energy / ((total_latency + eps) ** 0.8)
    energy_eff_norm = robust_normalize(energy_eff_ratio)
    
    # Dynamic starvation threshold: only boost tasks waiting longer than 25%-ile
    wait_threshold = np.percentile(ready_wait_time, 25) + eps
    wait_eligible = (ready_wait_time > wait_threshold).astype(float)
    # sqrt-scaling: smooth, bounded growth vs log1p — less sensitive to outliers, better for low-N cases
    wait_boost_raw = wait_eligible * np.sqrt(np.clip(ready_wait_time / (wait_threshold + eps), 0.0, 100.0))
    wait_boost = robust_normalize(wait_boost_raw)
    
    # Work and uncertainty: normalized but down-weighted to avoid overriding deadline signals
    work_norm = robust_normalize(remaining_work)
    uncertainty_norm = robust_normalize(uncertainty)
    
    # Hard deadline gating: zero energy weight if *any* task violates effective_slack
    has_violation = np.any(effective_slack < 0)
    energy_weight = 0.0 if has_violation else 0.9
    
    # Final score: smaller = higher priority
    # Strong deadline risk penalty, strong criticality reward, conditional energy optimization,
    # moderate latency & starvation terms, light uncertainty/work regularization
    score = (
        +3.8 * deadline_risk 
        - 2.4 * upward_rank_norm 
        + energy_weight * energy_eff_norm 
        + 0.4 * total_latency_norm 
        + 0.7 * wait_boost 
        + 0.05 * work_norm 
        + 0.05 * uncertainty_norm
    )
    
    # Ensure finite output: replace NaN/inf with large finite bounds
    return np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
