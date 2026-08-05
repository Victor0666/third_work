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
    v2 priority rule: Restored robust urgency + hardened DDL compliance + adaptive fairness.
    Key improvements:
    - Reintroduces trimmed-mean slack centrality for stable urgency scaling under small N
    - Combines crisp hard-DDL bias (for negative slack) with smooth sigmoid urgency (for near-deadline)
    - Strengthens fairness: wait-per-work gating now uses robust z-score AND work-quantile threshold
    - Adds critical-path energy penalty: scales energy cost by upward_rank only when slack is tight
    - Introduces uncertainty-aware slack normalization to suppress high-risk tasks near deadline
    - All divisions guarded; NaN/inf handled deterministically; deterministic clipping applied
    """
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=float).copy()
    min_comm_time = np.asarray(min_comm_time, dtype=float).copy()
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float).copy()
    slack = np.asarray(slack, dtype=float).copy()
    upward_rank = np.asarray(upward_rank, dtype=float).copy()
    remaining_work = np.asarray(remaining_work, dtype=float).copy()
    ready_wait_time = np.asarray(ready_wait_time, dtype=float).copy()
    uncertainty = np.asarray(uncertainty, dtype=float).copy()
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)

    def robust_minmax(x):
        if x.size == 0:
            return np.zeros_like(x)
        x_sorted = np.sort(x)
        trim_n = max(1, int(0.1 * len(x_sorted)))
        x_trimmed = x_sorted[trim_n:-trim_n] if len(x_sorted) > 2 * trim_n else x_sorted
        x_min, x_max = (np.min(x_trimmed), np.max(x_trimmed))
        rng = x_max - x_min + eps
        return np.clip((x - x_min) / rng, 0.0, 1.0)

    # Task duration baseline (execution + communication), with eps guard
    task_duration = min_exec_time + min_comm_time + eps

    # Robust relative slack: avoid inf/nan via safe division and fallback
    rel_slack = np.divide(slack, task_duration, out=np.full_like(slack, 0.0), where=task_duration != 0)
    rel_slack = np.where(np.isfinite(rel_slack), rel_slack, 0.0)

    # Robust slack centrality: trimmed mean of finite rel_slack for stable urgency anchor
    finite_rel_slack = rel_slack[np.isfinite(rel_slack)]
    slack_central = np.mean(finite_rel_slack) if len(finite_rel_slack) > 0 else 0.0

    # Crisp hard-DDL bias: dominates ranking when slack ≤ 0
    has_negative_slack = (slack <= 0.0).astype(float)
    urgency_bias = np.where(has_negative_slack, -1.0, 0.0)

    # Smooth urgency: sigmoid centered at slack_central, steepened near deadline (6.0 slope)
    urgency_sigmoid = 1.0 / (1.0 + np.exp(6.0 * (rel_slack - slack_central)))
    # Blend crisp and smooth: full bias when slack≤0, otherwise smooth urgency
    urgency = np.where(has_negative_slack, 1.0, urgency_sigmoid)

    # Lateness penalty: normalized by median duration, capped to prevent explosion
    lateness_penalty = np.where(slack < 0.0, np.clip(-slack / (np.median(task_duration) + eps), 0.0, 4.0), 0.0)

    # Critical-path latency penalty: upward_rank × |lateness|, normalized
    cp_latency_penalty = upward_rank * np.maximum(0.0, -slack) / (task_duration + eps)

    # Energy per work: guard against zero remaining_work
    energy_per_work = np.divide(min_incremental_energy, remaining_work + eps,
                                out=np.full_like(min_incremental_energy, eps),
                                where=remaining_work + eps != 0)

    # Criticality-weighted energy: only amplified when slack is tight (rel_slack ≤ 0.25)
    tight_slack_mask = (rel_slack <= 0.25).astype(float)
    crit_weight_factor = 1.0 + upward_rank / (np.median(upward_rank + eps) + eps)
    base_crit_energy = energy_per_work * crit_weight_factor
    # Slack-scaled energy: penalize high-energy tasks more aggressively under pressure
    slack_scale_factor = 1.0 + np.maximum(0.0, -slack) / (task_duration + eps)
    crit_weighted_energy = base_crit_energy * slack_scale_factor * tight_slack_mask

    # Uncertainty boost: dur_uncertainty scaled only when slack is tight
    dur_uncertainty = np.divide(uncertainty, task_duration + eps,
                                 out=np.zeros_like(uncertainty),
                                 where=task_duration + eps != 0)
    uncertainty_boost = dur_uncertainty * tight_slack_mask

    # Fairness: wait-per-work with robust z-scoring AND work-threshold (top 80% work)
    wait_per_work = np.divide(ready_wait_time, remaining_work + eps,
                              out=np.full_like(ready_wait_time, 0.0),
                              where=remaining_work + eps != 0)
    work_threshold = np.quantile(remaining_work, 0.2, method='midpoint') + eps
    wait_median = np.median(wait_per_work)
    wait_std = np.std(wait_per_work) + eps
    wait_z_threshold = wait_median + 1.5 * wait_std
    # Fairness gate requires both: non-trivial work AND high relative wait
    wait_gate = ((remaining_work >= work_threshold).astype(float) *
                 (wait_per_work > wait_z_threshold).astype(float))
    norm_wait_time = robust_minmax(ready_wait_time)
    wait_boost = norm_wait_time * wait_gate * 0.25

    # Normalize components robustly
    norm_urgency = robust_minmax(urgency)
    norm_lateness = robust_minmax(lateness_penalty)
    norm_cp_latency = robust_minmax(cp_latency_penalty)
    norm_energy = robust_minmax(crit_weighted_energy)
    norm_uncertainty = robust_minmax(uncertainty_boost)
    norm_upward = robust_minmax(upward_rank)

    # Hierarchical score: urgency dominates (negative weight), then latency, energy, etc.
    # Urgency term is subtracted to prioritize higher urgency → lower score
    score = (-2.5 * norm_urgency +
             1.3 * norm_lateness +
             1.0 * norm_energy +
             0.3 * norm_uncertainty -
             0.4 * norm_upward +
             0.25 * wait_boost +
             0.8 * norm_cp_latency +
             urgency_bias)  # crisp bias added outside norm for dominance

    # Final safeguard: clip and sanitize
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)

    return score
