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
    Self-evolved priority rule v2: Lexicographic deadline-criticality-energy dominance + starvation-aware fairness.
    
    Key evolutions:
    - Replaces convex weighting with *lexicographic dominance* (slack → cp_leverage → energy) to guarantee hard DDL compliance
    - Restores urgency discrimination via robust normalized slack ranking (trimmed quantile, not linear penalty)
    - Revives starvation boost for *all* tasks with high wait_ratio, but *gated by slack tolerance* (rel_slack > -0.1) to avoid interfering with critical lateness recovery
    - Introduces *critical-path latency leverage*: upward_rank * (1 + max(0,-slack)/task_duration) → amplifies critical tasks under pressure without breaking monotonicity
    - Uses MAD-based robust normalization (not std) for all metrics to resist outliers in small-N ready sets
    - All divisions guarded; NaN/inf replaced deterministically; no unbounded ops; shape invariant
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

    # Robust trimmed normalization using MAD for outlier resilience
    def robust_mad_normalize(x):
        if x.size == 0:
            return np.zeros_like(x)
        x = x.astype(float)
        med = np.median(x)
        abs_dev = np.abs(x - med)
        mad = np.median(abs_dev) + eps
        # Trim extremes: keep only points within [med - 3*mads, med + 3*mads]
        mask = (abs_dev <= 3.0 * mad)
        x_clipped = np.where(mask, x, med)
        x_min = np.min(x_clipped)
        x_max = np.max(x_clipped)
        rng = x_max - x_min + eps
        return np.clip((x - x_min) / rng, 0.0, 1.0)

    task_duration = min_exec_time + min_comm_time + eps
    neg_slack = np.maximum(-slack, 0.0)
    rel_slack = np.divide(slack, task_duration, out=np.full_like(slack, np.inf), where=task_duration!=0)
    
    # Primary key: normalized slack — smallest (most negative) first → highest priority
    # Use robust rank: lower quantile = higher urgency; invert for min-score priority
    if N == 1:
        norm_slack = np.array([0.0])
    else:
        # Rank-based: smaller slack → smaller rank index → smaller score
        # Use descending sort order so argmin picks most urgent
        sorted_indices = np.argsort(slack)
        ranks = np.empty_like(sorted_indices, dtype=float)
        ranks[sorted_indices] = np.arange(N, dtype=float)
        norm_slack = ranks / (N - 1 + eps) if N > 1 else np.array([0.0])

    # Secondary key: critical-path leverage — higher upward_rank per work → higher priority when slack is comparable
    cp_leverage = upward_rank / (remaining_work + eps)
    # Boost leverage under deadline pressure: amplify critical tasks when slack < 0
    cp_leverage_pressure = cp_leverage * (1.0 + np.clip(neg_slack / (task_duration + eps), 0.0, 2.0))
    norm_cp = robust_mad_normalize(cp_leverage_pressure)

    # Tertiary key: risk-weighted energy efficiency — lower energy_per_work → higher priority
    energy_per_work = np.divide(min_incremental_energy, remaining_work + eps, out=np.full_like(min_incremental_energy, 1e6), where=(remaining_work + eps)!=0)
    # Only scale up energy cost under true lateness (slack < -0.1s), not just rel_slack noise
    slack_energy_gate = (slack < -0.1).astype(float)
    energy_scale = 1.0 + slack_energy_gate * np.clip(neg_slack / (task_duration + eps), 0.0, 3.0)
    risk_energy = energy_per_work * energy_scale
    norm_energy = robust_mad_normalize(risk_energy)

    # Fairness: starvation boost activated when wait_ratio is extreme AND slack allows it (rel_slack > -0.1)
    wait_ratio = np.divide(ready_wait_time, remaining_work + eps, out=np.full_like(ready_wait_time, 0.0), where=(remaining_work + eps)!=0)
    wait_med = np.median(wait_ratio)
    wait_abs_dev = np.abs(wait_ratio - wait_med)
    wait_mad = np.median(wait_abs_dev) + eps
    wait_z = np.divide(wait_ratio - wait_med, wait_mad, out=np.zeros_like(wait_ratio), where=wait_mad!=0)
    starvation_gate = (rel_slack > -0.1).astype(float) * (wait_z > 1.75).astype(float)
    norm_wait = robust_mad_normalize(ready_wait_time)
    starvation_boost = norm_wait * starvation_gate * 0.18

    # Construct lexicographic composite: slack dominates, then cp, then energy, then fairness
    # Scale each level to avoid overflow and preserve ordering: 1e6 * primary + 1e3 * secondary + 1e0 * tertiary + boost
    primary_weight = 1e6
    secondary_weight = 1e3
    tertiary_weight = 1.0
    # Ensure all components are finite and bounded
    norm_slack = np.nan_to_num(norm_slack, nan=1.0, posinf=1.0, neginf=0.0)
    norm_cp = np.nan_to_num(norm_cp, nan=0.5, posinf=1.0, neginf=0.0)
    norm_energy = np.nan_to_num(norm_energy, nan=0.5, posinf=1.0, neginf=0.0)
    starvation_boost = np.nan_to_num(starvation_boost, nan=0.0, posinf=0.0, neginf=0.0)

    score = (
        primary_weight * norm_slack +
        secondary_weight * (1.0 - norm_cp) +  # higher cp → lower score
        tertiary_weight * norm_energy +
        starvation_boost
    )

    # Final guard: ensure finite, shape-(N,), deterministic
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
