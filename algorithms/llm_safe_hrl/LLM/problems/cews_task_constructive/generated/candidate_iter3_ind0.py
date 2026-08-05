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

    '''
    Self-evolved priority rule emphasizing *deadline feasibility robustness*, *risk-aware energy efficiency*, 
    and *fairness-aware starvation control*.

    Key advances over v1:
    - Replaces absolute slack normalization with *relative slack ratio*: slack / (min_exec_time + min_comm_time + eps)
      to capture urgency relative to task's intrinsic time cost — avoids distortion when deadlines vary widely.
    - Introduces *dynamic criticality-energy tradeoff*: uses upward_rank / (min_incremental_energy * (1 + uncertainty)^alpha),
      where alpha = clip(1.0 + 0.5 * max(0, -slack), 1.0, 3.0) — amplifies energy penalty under lateness risk.
    - Upgrades starvation control: *adaptive waiting penalty* scaled by remaining_work and slack, bounded [0, 0.25] 
      to prevent long-wait dominance when work is trivial or deadline is tight.
    - Adds *work-aware energy scaling*: min_incremental_energy is weighted by remaining_work / (median(remaining_work)+eps) 
      only for high-criticality tasks (upward_rank > median), making energy cost context-sensitive.
    - Uses *symmetric IQR normalization* with sign-preserving centering (median) and explicit finite bounds on all ratios.
    - All operations strictly eps-protected, nan/inf guarded, deterministic, and shape-compliant.
    '''
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)

    def robust_normalize(x):
        q1, q3 = np.quantile(x, [0.25, 0.75])
        iqr = q3 - q1 + eps
        center = np.median(x)
        return np.clip((x - center) / iqr, -10.0, 10.0)

    # Relative urgency: slack normalized by task's minimum time cost (exec + comm)
    task_min_duration = np.maximum(min_exec_time + min_comm_time, eps)
    rel_slack = slack / task_min_duration
    urgency_mask = (slack <= 0.0).astype(float)
    urgency_linear = np.maximum(0.0, -rel_slack) * 0.9
    urgency_soft = 1.0 / (1.0 + np.maximum(0.0, rel_slack) * 0.15 + eps)
    deadline_urgency = urgency_mask * (1.0 + urgency_linear) + (1.0 - urgency_mask) * urgency_soft

    # Dynamic risk exponent: higher penalty under lateness; bounded [1.0, 3.0]
    risk_exponent = np.clip(1.0 + 0.5 * np.maximum(0.0, -slack), 1.0, 3.0)
    energy_risk_weighted = min_incremental_energy * np.power(1.0 + uncertainty, risk_exponent)
    energy_safe = np.maximum(energy_risk_weighted, eps)
    
    # Criticality-energy ratio: clipped and robustly normalized
    crit_eff_ratio = upward_rank / energy_safe
    crit_eff_ratio = np.clip(crit_eff_ratio, 1e-06, 1e6)
    crit_eff_norm = robust_normalize(crit_eff_ratio)

    # Adaptive starvation penalty: scales with wait time but suppressed when remaining_work is low or slack is negative
    median_rw = np.median(remaining_work) + eps
    rw_ratio = np.clip(remaining_work / median_rw, 0.1, 10.0)
    wait_scale_factor = np.where(slack < 0.0, 0.5, np.where(rw_ratio < 0.5, 0.2, 1.0))
    max_wait = np.maximum(np.max(ready_wait_time), eps)
    wait_penalty = np.clip(ready_wait_time / max_wait, 0.0, 1.0) * wait_scale_factor * 0.25

    # Work-aware energy scaling: boost energy cost for high-criticality, high-work tasks
    median_ur = np.median(upward_rank) + eps
    ur_ratio = upward_rank / median_ur
    energy_work_weight = np.where(ur_ratio > 1.5, np.clip(rw_ratio, 1.0, 3.0), 1.0)
    energy_scaled = min_incremental_energy * energy_work_weight
    energy_norm = robust_normalize(energy_scaled)

    # Time cost and uncertainty normalization
    time_cost = np.sqrt(np.maximum(min_exec_time, eps)) + np.sqrt(np.maximum(min_comm_time, eps))
    time_norm = robust_normalize(time_cost)
    unc_norm = robust_normalize(uncertainty)
    work_norm = robust_normalize(remaining_work)

    # Final score: deadline urgency dominates; criticality-energy tradeoff second; others penalize
    score = (
        -3.2 * deadline_urgency
        - 1.6 * crit_eff_norm
        + 0.35 * time_norm
        + 0.28 * energy_norm
        + 0.22 * work_norm
        + 0.18 * unc_norm
        + wait_penalty
    )

    # Ensure finite, deterministic output with strict shape
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    return score
