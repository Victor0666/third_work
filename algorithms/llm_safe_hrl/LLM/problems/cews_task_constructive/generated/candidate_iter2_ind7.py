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
    Hybrid priority rule: combines Parent 2's smooth urgency & robust normalization
    with Parent 1's criticality-energy ratio and starvation-avoidance wait boost.
    
    Key improvements:
      - Uses sigmoid urgency (Parent 2) but anchors scale to median slack for physical stability
      - Replaces criticality-energy penalty with *criticality-per-energy* ratio (upward_rank / (energy + eps))
        to favor high-impact, low-energy tasks on critical paths — directly aligns with objective
      - Retains Parent 1's uncertainty-weighted wait boost but gated only for slack >= 0 (no lateness reward)
      - Introduces *slack-aware energy scaling*: energy_score is down-weighted when slack > median_slack
        (reducing energy focus when deadlines are relaxed), up-weighted when urgent
      - All terms normalized to [0,1] via robust min-max (Parent 2), then linearly combined with interpretable weights
      - Final score clipped and nan-cleaned for numerical safety
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
        """Min-max normalize to [0, 1]; handles constant arrays and avoids division by zero."""
        x_min, x_max = np.min(x), np.max(x)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x - x_min) / (x_max - x_min + eps)
    
    # Sigmoid urgency anchored to median slack (more stable than mean for skewed distributions)
    median_slack = np.median(slack)
    urgency_scale = np.abs(median_slack) + eps
    urgency = 1.0 / (1.0 + np.exp(-(slack - median_slack) / urgency_scale))
    
    # Criticality-per-energy ratio: higher = better (favor high-rank, low-energy tasks)
    crit_per_energy = upward_rank / (min_incremental_energy + eps)
    norm_crit_per_energy = robust_minmax_norm(crit_per_energy)
    
    # Energy score (lower is better), scaled by urgency: emphasize energy minimization when tight
    energy_score = robust_minmax_norm(min_incremental_energy)
    slack_normalized = (slack - median_slack) / (urgency_scale + eps)
    # Down-weight energy importance when slack > median; up-weight when slack < median
    energy_weight_factor = 1.0 + 0.5 * np.clip(-slack_normalized, 0.0, 1.0)
    weighted_energy_score = energy_score * energy_weight_factor
    
    # Execution & communication time scores (lower is better)
    exec_score = robust_minmax_norm(min_exec_time)
    comm_score = robust_minmax_norm(min_comm_time)
    
    # Waiting-time boost only for non-urgent tasks (slack >= 0) to prevent starvation
    wait_boost = np.where(slack >= 0, ready_wait_time, np.zeros_like(ready_wait_time))
    norm_wait_boost = robust_minmax_norm(wait_boost)
    
    # Uncertainty boost gated by urgency: only amplify priority for uncertain AND urgent tasks
    uncertainty_boost = uncertainty * (1.0 - urgency)  # Higher boost when urgency is low → avoid premature preemption
    norm_uncertainty_boost = robust_minmax_norm(uncertainty_boost)
    
    # Final score: smaller = higher priority
    # Weights sum to 1.0 for interpretability; deadline urgency dominates (0.4), energy tradeoff secondary (0.3)
    score = (
        0.4 * (1.0 - urgency) +                    # High urgency → low score (priority)
        0.3 * weighted_energy_score +             # Energy-aware, slack-modulated
        0.1 * exec_score +                        # Execution time penalty
        0.1 * comm_score +                        # Communication time penalty
        0.05 * (1.0 - norm_crit_per_energy) +     # Penalize low criticality-per-energy (i.e., high energy for low rank)
        0.03 * (1.0 - norm_wait_boost) +          # Boost long-waiting non-urgent tasks (starvation avoidance)
        0.02 * norm_uncertainty_boost             # Gentle nudge for uncertain-but-not-urgent tasks
    )
    
    # Ensure finite output and deterministic shape
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    return score
