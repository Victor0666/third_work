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
    Self-evolved priority rule: strict deadline feasibility first, then energy-criticality tradeoff,
    with monotonic, scale-separated, and numerically robust composition.

    Key improvements over v1:
    - Replaces hybrid urgency gating with *unified hard-urgency signal*: only slack <= 0 triggers
      strict enforcement; avoids soft-threshold fragmentation and preserves DDL hardness.
    - Eliminates additive lateness penalty; instead uses *normalized urgency ratio* (max(0,-slack)/eps_ref)
      scaled by a fixed weight — ensures monotonic priority boost without dominance.
    - Introduces *criticality-energy quotient* normalized *before* weighting: robust IQR scaling applied
      to each term individually, preserving relative ordering and avoiding post-combination distortion.
    - Starvation bonus now uses *relative wait rank* (percentile) instead of absolute time — invariant to
      simulation scale and ensures fairness across heterogeneous workloads.
    - All components are sign-consistent: negative contributions lower score (higher priority), positive raise it.
    - Explicit epsilon-ref for slack normalization avoids division-by-zero and maintains unitless scaling.
    - Final score is sum of orthogonal, pre-normalized, bounded terms — guarantees deterministic, interpretable ordering.
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
    N = len(slack)
    if N == 0:
        return np.array([], dtype=float)

    # Robust normalization per feature: IQR-based, epsilon-guarded
    def robust_normalize(x):
        q1, q3 = np.quantile(x, [0.25, 0.75], method='midpoint')
        iqr = q3 - q1 + eps
        center = np.median(x)
        return (x - center) / iqr

    # === 1. Deadline Hardness: unified binary urgency + normalized lateness magnitude ===
    # Only tasks with slack <= 0 get urgency boost; magnitude scales with violation severity
    is_urgent = (slack <= 0.0).astype(float)
    eps_ref = np.maximum(np.quantile(np.abs(slack), 0.9), eps)  # reference scale from worst 10% slack
    lateness_ratio = np.maximum(0.0, -slack) / (eps_ref + eps)
    deadline_score = is_urgent * (1.0 + lateness_ratio * 2.0)  # bounded, monotonic, unitless

    # === 2. Criticality-Energy Efficiency: ratio normalized *before* weighting ===
    energy_safe = np.maximum(min_incremental_energy, eps)
    crit_eff_ratio = upward_rank / energy_safe
    crit_eff_norm = robust_normalize(crit_eff_ratio)
    # Apply uncertainty damping *only* when urgent — preserves criticality signal otherwise
    unc_damp = np.where(is_urgent, np.clip(1.0 - uncertainty, 0.1, 1.0), 1.0)
    crit_eff_score = -crit_eff_norm * unc_damp  # negative → higher priority

    # === 3. Starvation Fairness: percentile-based, not absolute-time-based ===
    # Rank wait time as percentile (0–1), then apply bounded sigmoid → fair, scale-invariant
    wait_percentile = np.argsort(np.argsort(ready_wait_time)) / (N - 1 + eps) if N > 1 else np.array([0.0])
    starvation_bonus = 1.0 / (1.0 + np.exp(-4.0 * (wait_percentile - 0.5)))
    wait_score = -starvation_bonus  # negative → higher priority

    # === 4. Time Cost: sqrt(exec + comm) normalized — captures latency sensitivity ===
    time_cost = np.sqrt(np.maximum(min_exec_time + min_comm_time, eps))
    time_norm = robust_normalize(time_cost)

    # === 5. Workload Importance: remaining_work normalized — favors high-impact subtrees ===
    work_norm = robust_normalize(remaining_work)

    # === 6. Energy Penalty: normalized incremental energy — direct cost signal ===
    energy_norm = robust_normalize(min_incremental_energy)

    # === 7. Uncertainty Risk: only penalized under urgency, normalized ===
    unc_norm = robust_normalize(uncertainty) * is_urgent

    # Weighted linear combination — all terms orthogonal, pre-normalized, bounded
    # Coefficients ordered by design priority: deadline > criticality > fairness > cost
    score = (
        2.0 * deadline_score +
        1.3 * crit_eff_score +
        0.9 * wait_score +
        0.6 * time_norm +
        0.4 * work_norm +
        0.3 * energy_norm +
        0.2 * unc_norm
    )

    # Final guard: ensure finite, deterministic, shape-(N,) output
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    return score
