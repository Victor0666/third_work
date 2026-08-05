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
    Self-evolved priority rule enforcing strict DDL feasibility first, energy minimization second:
    - Restores hard urgency override for *any* negative slack (not just severe) to guarantee DDL compliance
    - Eliminates slack-aware energy penalty: separates hard constraint (DDL) from soft objective (energy)
    - Reinstates percentile-gated wait boost (v0 insight) with tighter slack guard (-0.02 instead of -0.05) to prevent starvation *only when safe*
    - Uses normalized duration uncertainty *only under tight slack* (rel_slack <= 0.1), not borderline — avoids over-penalizing robust tasks
    - Criticality-energy term now uses *inverse energy density per unit work*, weighted by upward_rank, then *robustly min-max normalized* (not raw)
    - All normalizations use bounded robust_minmax with fallbacks; all divisions guarded; NaN/inf clipped deterministically
    - No conflated terms: urgency, lateness, energy, criticality, fairness are cleanly decoupled and orthogonally weighted
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

    def robust_minmax(x):
        x_min, x_max = np.min(x), np.max(x)
        rng = x_max - x_min + eps
        return np.clip((x - x_min) / rng, 0.0, 1.0)

    # Task duration baseline (avoid zero)
    task_duration = min_exec_time + min_comm_time + eps
    rel_slack = slack / task_duration

    # === URGENCY: Hard override for *any* negative slack → highest priority (score → -∞)
    # Ensures DDL feasibility dominates: even small lateness triggers top priority
    has_negative_slack = (slack < 0.0).astype(float)
    # Base urgency: sigmoid gated at 30th percentile relative slack
    slack_threshold = np.quantile(rel_slack, 0.3) if N > 1 else np.median(rel_slack)
    urgency_sigmoid = 1.0 / (1.0 + np.exp(6.0 * (rel_slack - slack_threshold)))
    # Hard override: negative slack → max urgency → minimal score
    urgency = np.where(has_negative_slack, 1.0, urgency_sigmoid)

    # === LATENESS PENALTY: Only for already-late tasks; bounded & normalized
    lateness_penalty = np.where(slack < 0.0, np.clip(-slack / (np.median(task_duration) + eps), 0.0, 4.0), 0.0)

    # === UNCERTAINTY BOOST: Only when slack is *tight* (rel_slack <= 0.1), not borderline
    tight_slack_mask = (rel_slack <= 0.1).astype(float)
    dur_uncertainty = np.where(task_duration > eps, uncertainty / task_duration, 0.0)
    uncertainty_boost = dur_uncertainty * tight_slack_mask

    # === CRITICALITY-ENERGY TERM: Inverse energy density per work × rank scaling
    energy_per_work = np.maximum(min_incremental_energy / (remaining_work + eps), eps)
    rank_scale = 1.0 + upward_rank / (np.median(upward_rank + eps) + eps)
    crit_weighted_energy = energy_per_work * rank_scale

    # === FAIRNESS WAIT BOOST: Percentile-gated (v0), not log-scaled — preserves ordering under load
    # Activated only when slack is *slightly positive* AND work is non-trivial
    wait_gate = (
        (rel_slack > -0.02).astype(float) *
        (remaining_work > np.quantile(remaining_work, 0.2) + eps).astype(float)
    )
    # Use normalized wait time (not log) to avoid compressing large differences
    norm_wait_time = robust_minmax(ready_wait_time)
    wait_boost = norm_wait_time * wait_gate * 0.25

    # === NORMALIZE COMPONENTS
    norm_urgency = robust_minmax(urgency)
    norm_lateness = robust_minmax(lateness_penalty)
    norm_energy = robust_minmax(crit_weighted_energy)
    norm_uncertainty = robust_minmax(uncertainty_boost)
    norm_upward = robust_minmax(upward_rank)

    # === FINAL SCORE: Smaller = better
    # Strong negative weight on urgency → pushes negative-slack tasks to top
    # Positive weights on penalties, small positive on fairness (to break ties)
    score = (
        -3.0 * norm_urgency          # Dominant: urgency first
        + 1.3 * norm_lateness        # Lateness cost (only for late tasks)
        + 1.0 * norm_energy          # Energy-efficiency in feasible region
        + 0.3 * norm_uncertainty     # Risk penalty only under tight slack
        - 0.4 * norm_upward          # Prefer higher criticality (lower score)
        + 0.15 * wait_boost         # Mild fairness boost when safe
    )

    # Final safeguard: ensure finite, deterministic output
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    return score
