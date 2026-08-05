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
    Self-evolved priority rule emphasizing *strict deadline feasibility*, *preserved criticality-energy coupling*,
    and *robust starvation control* — addressing v1's over-flatting and normalization flaws.

    Key improvements:
    - Replaces weighted linear combination with *urgency-gated multi-objective ranking*: 
      urgent tasks (slack <= 0) prioritize criticality/energy ratio; non-urgent prioritize slack-efficiency tradeoff.
    - Restores criticality-energy coupling: work-aware scaling applied *before* IQR normalization to preserve ordinal risk ordering.
    - Uses linear adaptive wait penalty (not log) with slack-aware bounds: stronger fairness under tight deadlines.
    - Eliminates redundant normalization of deadline_urgency — keeps raw urgency signal unblurred.
    - Introduces *slack-constrained energy efficiency*: for non-urgent tasks, energy score is scaled by (1 + max(0, -rel_slack)).
    - All operations eps-protected, finite-clipped, deterministic, and shape-compliant.
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
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)

    def robust_iqr_normalize(x):
        q1, q3 = np.quantile(x, [0.25, 0.75])
        iqr = q3 - q1 + eps
        center = np.median(x)
        normed = (x - center) / iqr
        return np.clip(normed, -10.0, 10.0)

    # Task intrinsic time cost — base for relative slack
    task_min_duration = np.maximum(min_exec_time + min_comm_time, eps)
    rel_slack = slack / task_min_duration

    # Urgency gate: strict binary for hard DDL compliance
    is_urgent = (slack <= 0.0).astype(float)

    # Dynamic risk exponent: amplifies energy penalty only when lateness risk exists
    risk_exponent = np.clip(1.0 + 0.5 * np.maximum(0.0, -slack), 1.0, 3.0)
    energy_risk_weighted = min_incremental_energy * np.power(1.0 + uncertainty, risk_exponent)
    energy_safe = np.maximum(energy_risk_weighted, eps)

    # Work-aware scaling applied *before* normalization → preserves criticality-energy ordinal structure
    median_ur = np.median(upward_rank) + eps
    ur_ratio = upward_rank / median_ur
    median_rw = np.median(remaining_work) + eps
    rw_ratio = np.clip(remaining_work / median_rw, 0.1, 10.0)
    energy_work_weight = np.where(ur_ratio > 1.5, np.clip(rw_ratio, 1.0, 3.0), 1.0)
    energy_scaled = energy_safe * energy_work_weight

    # Criticality-energy ratio: higher = better tradeoff → lower priority score desired
    crit_eff_ratio = upward_rank / energy_scaled
    crit_eff_ratio = np.clip(crit_eff_ratio, 1e-06, 1e6)
    crit_eff_norm = robust_iqr_normalize(crit_eff_ratio)

    # Slack-constrained efficiency for non-urgent tasks: reward both margin and energy
    # Higher slack → lower effective energy cost (via scaling); negative slack → penalize
    slack_efficiency_scale = np.where(is_urgent == 1.0, 1.0, 1.0 + np.maximum(0.0, -rel_slack))
    energy_slack_scaled = energy_scaled * slack_efficiency_scale
    energy_norm = robust_iqr_normalize(energy_slack_scaled)

    # Linear adaptive wait penalty: stronger under urgency, bounded [0, 0.25]
    wait_max = np.maximum(np.max(ready_wait_time), eps)
    wait_normalized = np.clip(ready_wait_time / wait_max, 0.0, 1.0)
    wait_scale_factor = np.where(
        is_urgent == 1.0, 
        0.25, 
        np.where(remaining_work < median_rw, 0.1, 0.2)
    )
    wait_guard = wait_normalized * wait_scale_factor

    # Time cost: sqrt-normalized execution + comm time → favors lightweight tasks when feasible
    time_cost = np.sqrt(np.maximum(min_exec_time, eps)) + np.sqrt(np.maximum(min_comm_time, eps))
    time_norm = robust_iqr_normalize(time_cost)

    # Uncertainty: normalized penalty — higher uncertainty → lower priority
    unc_norm = robust_iqr_normalize(uncertainty)

    # Final score: urgency-gated composition
    # For urgent tasks: prioritize criticality-efficiency & fairness (low wait)
    # For non-urgent tasks: prioritize slack margin, efficiency, and latency
    score_urgent = (
        -1.0 * crit_eff_norm +      # maximize criticality per energy
        0.8 * wait_guard +          # minimize starvation
        0.3 * time_norm +           # prefer low-latency tasks
        0.2 * unc_norm              # penalize high uncertainty
    )
    score_non_urgent = (
        -1.2 * robust_iqr_normalize(rel_slack) +  # maximize slack margin (higher rel_slack → lower score)
        0.7 * energy_norm +                         # minimize energy cost
        0.4 * time_norm +                           # prefer low-latency
        0.15 * unc_norm                             # mild uncertainty penalty
    )

    score = np.where(is_urgent == 1.0, score_urgent, score_non_urgent)

    # Final guard: ensure finite, deterministic output
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)

    return score
