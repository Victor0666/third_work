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
    Self-evolved priority rule v2: Strict deadline-first enforcement with orthogonal, decoupled terms,
    starvation-aware fairness, and *term-wise robust normalization* (not cross-term) to preserve ordinal stability.
    
    Key improvements over v1:
      - Eliminated redundant energy_latency_risk term → removes coupling between duration & uncertainty under safe slack
      - Replaced cross-term robust_minmax with *per-term bounded normalization* using quantile-based scaling
        to prevent rank inversion under constraint pressure; preserves relative ordering within each dimension
      - Unified urgency model: hard override for slack < 0, linear ramp for -0.1 <= slack < 0, sigmoid elsewhere
      - Introduced *work-normalized criticality*: upward_rank / (remaining_work + eps), scaled by energy density
      - Wait boost now uses *relative wait percentile* (not absolute time) + strict safety guard (rel_slack > -0.01)
      - All divisions guarded; NaN/inf replaced deterministically before normalization; no term dominates via clipping
      - Final score is convex combination of normalized components — no negative weights → ensures monotonic priority semantics
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

    # Precompute safe denominators and clipped values
    task_duration = np.clip(min_exec_time + min_comm_time, eps, np.inf)
    rel_slack = np.divide(slack, task_duration, out=np.full_like(slack, np.nan), where=task_duration != 0)
    rel_slack = np.nan_to_num(rel_slack, nan=0.0, posinf=1e6, neginf=-1e6)

    # === URGENCY: hard first, then graded ramp, then sigmoid ===
    has_negative_slack = (slack < 0.0).astype(float)
    urgency = np.zeros_like(slack)
    # Hard override
    urgency = np.where(has_negative_slack, 1.0, urgency)
    # Linear ramp for near-deadline (-0.1 ≤ slack < 0)
    in_ramp = np.logical_and(slack >= -0.1, slack < 0.0)
    urgency = np.where(in_ramp, 1.0 - (slack / 0.1), urgency)
    # Sigmoid for slack ≥ -0.1
    sigmoid_region = (rel_slack >= -0.1)
    base_sigmoid = 1.0 / (1.0 + np.exp(-4.0 * (rel_slack + 0.1)))
    urgency = np.where(sigmoid_region & ~has_negative_slack & ~in_ramp, base_sigmoid, urgency)

    # === LATENESS PENALTY: only active when slack < 0, capped at 5x median duration ===
    lateness_penalty = np.where(slack < 0.0,
                                np.clip(-slack / (np.median(task_duration) + eps), 0.0, 5.0),
                                0.0)

    # === CRITICALITY-AWARE ENERGY DENSITY: upward_rank per unit work × energy per work ===
    work_norm_rank = np.divide(upward_rank, remaining_work + eps, out=np.zeros_like(upward_rank), where=(remaining_work + eps) != 0)
    energy_per_work = np.divide(min_incremental_energy, remaining_work + eps, out=np.zeros_like(min_incremental_energy), where=(remaining_work + eps) != 0)
    crit_energy_density = np.clip(work_norm_rank * energy_per_work, eps, np.inf)

    # === UNCERTAINTY MODULATION: only under tight slack (rel_slack ≤ 0.05), normalized per-task ===
    tight_slack_mask = (rel_slack <= 0.05).astype(float)
    dur_uncertainty = np.divide(uncertainty, task_duration, out=np.zeros_like(uncertainty), where=task_duration != 0)
    uncertainty_boost = dur_uncertainty * tight_slack_mask

    # === STARVATION AWARENESS: relative wait percentile, gated strictly by slack safety ===
    wait_percentile = np.zeros_like(ready_wait_time)
    if N > 1:
        wait_sorted = np.sort(ready_wait_time)
        ranks = np.searchsorted(wait_sorted, ready_wait_time, side='left')
        wait_percentile = (ranks.astype(float) + 1.0) / float(N)
    else:
        wait_percentile = np.array([0.5])
    safe_slack_mask = (rel_slack > -0.01).astype(float)
    wait_boost = wait_percentile * safe_slack_mask * 0.3

    # === TERM-WISE ROBUST NORMALIZATION (quantile-based, not min-max) to preserve ordinal stability ===
    def quantile_normalize(x, q_low=0.1, q_high=0.9):
        x = np.nan_to_num(x, nan=np.median(x), posinf=np.median(x), neginf=np.median(x))
        q_lo = np.quantile(x, q_low) if len(x) > 1 else np.median(x)
        q_hi = np.quantile(x, q_high) if len(x) > 1 else np.median(x)
        rng = q_hi - q_lo + eps
        return np.clip((x - q_lo) / rng, 0.0, 1.0)

    norm_urgency = quantile_normalize(urgency)
    norm_lateness = quantile_normalize(lateness_penalty)
    norm_crit_energy = quantile_normalize(crit_energy_density)
    norm_uncertainty = quantile_normalize(uncertainty_boost)
    norm_wait = quantile_normalize(wait_boost)

    # === FINAL SCORE: convex combination (all weights ≥ 0) → monotonic, stable, interpretable ===
    # Prioritizes urgency first, then penalizes lateness, then energy/criticality, then uncertainty/starvation
    score = (
        0.45 * norm_urgency +
        0.25 * norm_lateness +
        0.15 * norm_crit_energy +
        0.08 * norm_uncertainty +
        0.07 * norm_wait
    )

    # Ensure finite output, deterministic shape
    score = np.nan_to_num(score, nan=1.0, posinf=1.0, neginf=0.0)
    score = np.clip(score, 0.0, 1.0)
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
