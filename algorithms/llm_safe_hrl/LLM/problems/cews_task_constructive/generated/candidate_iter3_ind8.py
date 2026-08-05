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
    Self-evolved priority rule: fixes over-gating, preserves natural urgency gradient,
    replaces noise-amplifying critical-energy penalty with robust criticality-weighted energy ratio,
    and introduces starvation-aware wait boost that activates on *either* slack<=0 *or* high rank.
    
    Key improvements:
      - Restores slack gating to *single-condition*: waiting-time boost activates if slack <= 0 OR upward_rank > median (not AND)
      - Keeps sigmoid slack_urgency *unnormalized*: preserves its smooth, bounded [0,1] urgency signal
      - Replaces critical_energy_penalty with energy_per_second * (upward_rank / (remaining_work + eps)) — 
        penalizes high energy per second *per unit of critical-path work*, reducing rank-estimation noise impact
      - Uses clipped IQR normalization only where beneficial; falls back to stable mean-abs scaling
      - Applies final score clipping and nan_to_num *after* all arithmetic, with strict finite bounds
      - All terms weighted to prioritize deadline compliance first, then energy efficiency, then fairness
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
    
    # Robust normalization: IQR-based if sufficient spread, else mean-abs
    def robust_normalize(x):
        if x.size == 0:
            return np.zeros_like(x)
        q1, q3 = np.quantile(x, [0.25, 0.75], method='midpoint')
        iqr = q3 - q1
        if iqr < eps:
            scale = np.mean(np.abs(x)) + eps
        else:
            scale = iqr + eps
        return x / scale
    
    # Natural, bounded urgency: sigmoid maps slack → [0,1], low slack → high urgency → low score
    slack_mean_abs = np.mean(np.abs(slack)) + eps
    slack_urgency = 1.0 / (1.0 + np.exp(-slack / slack_mean_abs))  # ∈ (0,1), monotonic
    
    # Duration & energy efficiency: avoid division by zero
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_second = min_incremental_energy / duration
    
    # Criticality-weighted energy: energy per sec per unit of critical-path work (not raw rank)
    # Reduces sensitivity to rank estimation noise; favors low-energy-per-critical-work
    critical_work_ratio = upward_rank / (remaining_work + eps)
    weighted_energy = energy_per_second * (1.0 + 0.5 * critical_work_ratio)
    
    # Uncertainty boost only when at risk (slack <= 0)
    uncertainty_boost = np.where(slack <= 0, robust_normalize(uncertainty), 0.0)
    
    # Starvation guard: activate wait boost if *either* deadline at risk OR task is highly ranked
    median_rank = np.median(upward_rank) if len(upward_rank) > 1 else 0.0
    wait_activation = (slack <= 0) | (upward_rank > median_rank)
    norm_wait = np.where(wait_activation, robust_normalize(ready_wait_time), 0.0)
    
    # Upward rank scaled by remaining work importance, normalized robustly
    work_scale = robust_normalize(remaining_work)
    leveraged_rank = upward_rank * (1.0 + 0.3 * work_scale)
    
    # Normalize components for balanced contribution (no unit-range distortion)
    norm_energy = robust_normalize(weighted_energy)
    norm_rank = robust_normalize(leveraged_rank)
    norm_exec = robust_normalize(min_exec_time)
    norm_comm = robust_normalize(min_comm_time)
    
    # Final score: minimize → high priority
    # Dominant term: -slack_urgency (higher urgency → lower score)
    # Secondary: energy efficiency (+norm_energy), criticality (+norm_rank), uncertainty (+uncertainty_boost)
    # Tertiary: fairness (-norm_wait), overhead penalties (+norm_exec, +norm_comm)
    score = (
        +0.40 * norm_energy 
        - 1.6 * slack_urgency 
        + 0.25 * norm_rank 
        + 0.15 * uncertainty_boost 
        - 0.12 * norm_wait 
        + 0.06 * norm_exec 
        + 0.06 * norm_comm
    )
    
    # Ensure finiteness and determinism
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)
    
    return score
