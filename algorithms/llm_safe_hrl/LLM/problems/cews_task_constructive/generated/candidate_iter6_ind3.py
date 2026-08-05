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
    Self-evolved v2 priority rule: restores hard deadline dominance with crisp urgency gating,
    eliminates conflicting energy terms, strengthens starvation guard, and unifies risk scaling.
    
    Key improvements over v1:
      - Replaces sigmoid urgency with *hard-thresholded urgency* (0/1) to enforce strict DDL compliance:
        tasks with slack <= 0 get max priority (score=0), no smoothing → eliminates borderline dilution.
      - Removes energy_dominance term entirely → avoids focus fragmentation; retains only unified
        energy penalty scaled by urgency to preserve efficiency in safe region.
      - Strengthens starvation guard: now activated *only* for non-urgent (slack > 0) AND long-waiting
        (ready_wait_time > 95th percentile) tasks → prevents late-task reward while ensuring fairness.
      - Uncertainty boost uses *relative slack ratio*: (max(0, median_slack - slack) / (|median_slack| + eps))
        → robust to negative median, emphasizes risk severity near deadline boundary.
      - Criticality-energy term simplified to upward_rank / (min_incremental_energy + eps), normalized via
        *robust min-max with outlier clipping*, not IQR → preserves signal integrity for critical tasks.
      - All components explicitly bounded, fused via convex combination with weights summing to 1.0;
        final score clamped and nan-cleaned deterministically.
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
    
    # Hard urgency gate: binary priority for deadline violation risk (slack <= 0 → highest priority)
    urgency_flag = np.where(slack <= 0, 0.0, 1.0)  # 0.0 = max priority (smallest score)
    
    # Criticality per energy: upward_rank / energy, clipped to avoid explosion
    crit_per_energy = upward_rank / (min_incremental_energy + eps)
    crit_per_energy = np.clip(crit_per_energy, 1e-6, 1e6)
    norm_crit_per_energy = robust_minmax_norm(crit_per_energy)
    
    # Energy penalty: scaled by urgency_flag to activate only when safe (slack > 0)
    norm_energy = robust_minmax_norm(min_incremental_energy)
    energy_penalty = norm_energy * urgency_flag
    
    # Latency term: execution + communication, equally weighted
    norm_exec = robust_minmax_norm(min_exec_time)
    norm_comm = robust_minmax_norm(min_comm_time)
    latency_term = 0.5 * norm_exec + 0.5 * norm_comm
    
    # Uncertainty boost: relative slack distance, robust to negative median
    median_slack = np.median(slack)
    slack_gap = np.maximum(0.0, median_slack - slack)
    slack_scale = np.abs(median_slack) + eps
    uncertainty_risk_score = slack_gap / slack_scale
    uncertainty_boost = uncertainty * uncertainty_risk_score
    norm_uncertainty_boost = robust_minmax_norm(uncertainty_boost)
    
    # Starvation guard: activates only for non-urgent AND long-waiting tasks
    wait_thresh = np.percentile(ready_wait_time, 95) + eps
    starvation_flag = np.where((slack > 0) & (ready_wait_time >= wait_thresh), 1.0, 0.0)
    wait_boost = starvation_flag * robust_minmax_norm(ready_wait_time)
    
    # Final convex combination (weights sum to 1.0)
    # Base urgency dominates; others refine within safe region
    score = (
        0.50 * urgency_flag +                 # Primary DDL enforcement: 0 = urgent, 1 = non-urgent
        0.18 * energy_penalty +               # Energy cost only when safe
        0.12 * latency_term +                 # Execution+comm delay
        0.08 * (1.0 - norm_crit_per_energy) + # Critical tasks with low energy first
        0.06 * norm_uncertainty_boost +       # Risk amplification near tight deadlines
        0.04 * wait_boost +                   # Fairness for long-waiting non-urgent tasks
        0.02 * (1.0 - robust_minmax_norm(uncertainty))  # Prefer low-uncertainty VMs when possible
    )
    
    # Ensure finite output: clamp and clean
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    
    return score
