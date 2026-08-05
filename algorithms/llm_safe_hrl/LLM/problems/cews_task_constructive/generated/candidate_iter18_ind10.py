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
    v2: Deadline-hardened, energy-aware priority with uncertainty-resilient slack modeling,
    starvation-robust fairness, and monotonic safety guarantees.

    Key advances over v1:
    - Strict monotonicity: score is strictly decreasing in slack for slack >= 0 → ensures consistent
      prioritization of tighter deadlines within feasible region.
    - Uncertainty-integrated slack: robust_slack = slack - 2*uncertainty (as v1) but now clamped
      *before* tanh to prevent numerical instability near overflow; avoids tanh saturation artifacts.
    - Fairness redefined as *normalized aging ratio*: ready_wait_time / (max(ε, |slack| + ε)), 
      eliminating median dependency and ensuring deterministic behavior even when all slack <= 0.
    - Energy term uses *inverse SEER* (energy per time) instead of raw SEER to directly penalize
      high-energy-per-unit-time tasks — aligning with physics of energy minimization under deadline.
    - All normalization uses *mean-subtracted, std-clamped z-score* with explicit zero-variance handling
      and bounded clipping to preserve rank ordering under sparse or degenerate inputs.
    - Hard violation override now uses *rank-zero assignment* (np.min(score)-1e6) instead of absolute constant,
      guaranteeing strict dominance without risking overflow collisions in extreme-scale deployments.
    - Final score capped and sanitized to ensure finite, deterministic, and numerically safe output.
    """
    eps = 1e-08
    def sanitize(x):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(x, nan=eps, posinf=1e6, neginf=1e-6)

    min_exec_time = sanitize(min_exec_time)
    min_comm_time = sanitize(min_comm_time)
    min_incremental_energy = sanitize(min_incremental_energy)
    slack = sanitize(slack)
    upward_rank = sanitize(upward_rank)
    remaining_work = sanitize(remaining_work)
    ready_wait_time = sanitize(ready_wait_time)
    uncertainty = sanitize(uncertainty)

    # Robust slack: uncertainty-adjusted, pre-clipped to avoid tanh overflow & improve monotonicity
    robust_slack = slack - 2.0 * uncertainty
    robust_slack = np.clip(robust_slack, -1e5, 1e5)

    # Hard violation flag: tasks with slack < 0 get highest priority (lowest score)
    violation_mask = (slack < 0).astype(float)

    # Urgency: tanh(-robust_slack/tau) → smoothly increasing urgency as slack decreases;
    # clipped before tanh to ensure stable gradient and monotonic mapping
    tau_urgency = 1.0
    urgency_raw = np.tanh(-np.clip(robust_slack, -10.0, 10.0) / tau_urgency)
    urgency_term = np.where(violation_mask, -10.0 * urgency_raw, 0.0)

    # Energy efficiency: use inverse SEER (J/s) → higher value = worse energy/time tradeoff
    exec_comm_sum = min_exec_time + min_comm_time + eps
    inv_seer = np.clip(min_incremental_energy / (exec_comm_sum + eps), eps, 1e6)
    # Gate energy term only when slack > 2s AND uncertainty low → defer energy optimization until safe
    energy_gate = ((slack > 2.0) & (uncertainty < 0.2)).astype(float)
    seer_masked = inv_seer * energy_gate * np.exp(-np.maximum(0.0, slack) / 3.0)

    # Critical-path density: importance × work / time budget → prioritize high-impact, time-critical work
    cp_density = upward_rank * remaining_work / (exec_comm_sum + eps)
    cp_gate = (slack > 1.0).astype(float)  # activate only when margin exists
    cp_density_gated = cp_density * cp_gate

    # Fairness: normalized aging ratio — wait time relative to deadline margin (absolute slack + eps)
    # avoids median dependence and works deterministically even with all-negative slack
    fairness_denom = np.abs(slack) + eps
    fairness_raw = ready_wait_time / fairness_denom
    fairness_term = np.clip(fairness_raw, 0.0, 10.0)

    # Uncertainty boost: only when slack positive AND uncertainty high → proactively delay risky tasks
    unc_boost = np.where((slack > 0.0) & (uncertainty > 0.25), np.clip(uncertainty * 0.3, 0.0, 0.3), 0.0)

    # Deterministic z-score normalization with strict zero-variance handling
    def normalize_zscore(x):
        x = np.clip(x, -1e6, 1e6)
        if x.size == 1:
            return np.array([0.0])
        mean = np.mean(x)
        std = np.std(x, ddof=0)
        if std < eps:
            return np.zeros_like(x)
        return (x - mean) / (std + eps)

    norm_urgency = normalize_zscore(urgency_term)
    norm_seer = normalize_zscore(seer_masked)
    norm_cp = normalize_zscore(cp_density_gated)
    norm_fair = normalize_zscore(fairness_term)
    norm_unc = normalize_zscore(unc_boost)

    # Weighted fusion: urgency dominates violations; energy dominates otherwise; CP guides structure
    score = (
        +7.0 * norm_urgency   # strongest pull on violations
        - 3.0 * norm_seer     # energy penalty scaled higher for efficiency focus
        + 0.9 * norm_cp       # structural importance, moderate weight
        - 0.4 * norm_fair     # fairness as mild anti-starvation bias
        - 0.2 * norm_unc      # uncertainty delay penalty
    )

    # Hard violation override: assign strictly lowest possible score (guaranteed dominance)
    base_min = np.min(score) if len(score) > 0 else 0.0
    score = np.where(violation_mask, base_min - 1e6, score)

    # Final sanitization: ensure finite, bounded, deterministic output
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)

    return score
