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
    Self-evolved priority rule: balances hard-DDL safety, energy efficiency, and starvation avoidance
    with calibrated weights, adaptive thresholds, and decoupled uncertainty handling.
    
    Key improvements over v1:
    - Reduced deadline risk weight (-2.5) to avoid noise amplification; retains strong penalty floor
    - Replaced percentile wait-threshold with dynamic, workload-relative threshold (0.05 * median_latency)
      ensuring aging activates even under low-variance waiting
    - Uncertainty urgency now directly scaled by |slack|^{-1} for near-deadline tasks (not slack-safe),
      restoring urgency when deadlines are tight
    - Energy efficiency term modulated by upward_rank *only* for high-criticality tasks (top 30%),
      avoiding dilution on trivial paths
    - Added robustness: all normalizations guard against zero/nan via eps and fallback scaling
    - Final score clipped to strict finite bounds to prevent overflow in downstream argmin
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
        q1, q3 = np.quantile(x, [0.25, 0.75], method='midpoint')
        iqr = q3 - q1
        # Fallback to mean absolute deviation if IQR near zero
        scale = iqr if iqr > eps else np.mean(np.abs(x - np.mean(x))) + eps
        return x / (scale + eps)
    
    # Deadline risk: bounded sigmoid + hard linear penalty for negative slack (< -1ms)
    slack_abs_med = np.abs(np.median(slack)) + eps
    sigmoid_risk = 1.0 / (1.0 + np.exp(-slack / slack_abs_med))
    hard_penalty = np.where(slack < -eps, -slack * 5.0, 0.0)  # Softer than v1 but deterministic
    deadline_risk = sigmoid_risk + hard_penalty
    
    # Latency pressure (execution + comm) — normalized for scheduling urgency
    latency_pressure = min_exec_time + min_comm_time
    norm_latency = robust_normalize(latency_pressure)
    
    # Energy efficiency: marginal energy per unit remaining work, but only weighted for critical paths
    eff_ratio = min_incremental_energy / (remaining_work + eps)
    ur_thresh = np.quantile(upward_rank, 0.7, method='midpoint')  # top 30% criticality
    critical_mask = (upward_rank >= ur_thresh).astype(float)
    weighted_eff_ratio = eff_ratio * critical_mask
    norm_eff_ratio = robust_normalize(weighted_eff_ratio)
    
    # Upward rank contribution: only for high-importance nodes, normalized and inverted (higher rank → lower score)
    norm_upward = robust_normalize(np.where(upward_rank >= ur_thresh, upward_rank, 0.0))
    
    # Ready-wait boost: dynamic threshold = 5% of median latency pressure (robust across scales)
    median_latency = np.median(latency_pressure) + eps
    wait_threshold = 0.05 * median_latency
    waiting_boost = np.where(ready_wait_time > wait_threshold, ready_wait_time, 0.0)
    norm_waiting = robust_normalize(waiting_boost)
    
    # Uncertainty urgency: inversely proportional to |slack| for urgency amplification near deadline
    # Avoids division by zero: clamp slack magnitude to [eps, inf]
    slack_mag = np.abs(slack) + eps
    uncertainty_urgency = uncertainty / slack_mag
    norm_uncertainty = robust_normalize(uncertainty_urgency)
    
    # Final score: smaller = higher priority
    # Weights tuned to favor deadline compliance first, then energy on critical paths, then fairness
    score = (
        -2.5 * robust_normalize(deadline_risk)   # Strong but stable deadline enforcement
        + 0.2 * norm_latency                      # Prefer low-latency tasks when safe
        + 0.4 * norm_eff_ratio                    # Higher weight on energy-efficiency for critical paths
        + 0.12 * norm_uncertainty                 # Uncertainty matters most when slack is tiny
        + 0.13 * norm_waiting                     # Fairness: aging kicks in early & reliably
        - 0.15 * norm_upward                      # Encourage critical-path progress
    )
    
    # Ensure finite output: clip extremes and replace NaN/inf
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    
    return score
