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
    Self-evolved priority rule v2: addresses over-gating and z-score noise by:
      - Replacing relative-slack gating with *absolute slack thresholding* (robust to duration skew)
      - Using *multiplicative urgency-latency coupling*: lateness penalty modulates urgency base, not additive
      - Switching to *rank-based normalization* (percentile) for stability under low-N and sparse-ready regimes
      - Introducing *deadline-aware energy fairness*: energy term activated only when slack > 0, scaled by sigmoid(slack/median_duration)
      - Adding *critical-path starvation boost*: wait time weighted by upward_rank and normalized by remaining_work to prevent bias toward leaf nodes
      - Removing all z-score dependencies; using clipped percentile ranks (0–1) for monotonic, bounded, interpretable features
      - Enforcing strict finite output via double-clipping and nan-to-num with conservative bounds
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

    # Compute robust base metrics
    task_duration = np.maximum(min_exec_time + min_comm_time, eps)
    median_duration = np.median(task_duration) + eps

    # Absolute slack threshold: urgent if slack < 0.5 * median_duration (not relative)
    is_urgent = (slack < 0.5 * median_duration).astype(float)
    
    # Multiplicative urgency-latency coupling: urgency base amplified by lateness risk
    urgency_base = np.where(
        is_urgent,
        1.0 + np.maximum(0.0, -slack) / (median_duration + eps),  # linear penalty per sec
        1.0 / (1.0 + np.maximum(0.0, slack) / (median_duration + eps) + eps)
    )
    lateness_amplifier = 1.0 + np.clip(uncertainty, 0.0, 2.0) * 3.0
    urgency_score = urgency_base * lateness_amplifier

    # Deadline-aware energy fairness: only penalize energy when slack > 0, smoothly activated
    energy_activation = 1.0 / (1.0 + np.exp(-np.clip(slack / (median_duration + eps), -6.0, 6.0)))
    risk_adjusted_energy = min_incremental_energy * (1.0 + 0.5 * np.clip(uncertainty, 0.0, 1.0))
    energy_score = risk_adjusted_energy * energy_activation

    # Latency-criticality synergy: upward_rank × duration / (energy + eps), capped
    synergy_num = upward_rank * task_duration * (1.0 + np.clip(uncertainty, 0.0, 1.0))
    synergy_denom = np.maximum(risk_adjusted_energy, eps)
    latency_criticality_synergy = np.divide(
        synergy_num, synergy_denom,
        out=np.full(N, 1e-6), where=synergy_denom != 0
    )
    latency_criticality_synergy = np.clip(latency_criticality_synergy, 1e-6, 1e6)

    # Critical-path starvation boost: only when slack > 0 and upward_rank > 0
    # Normalized by remaining_work to avoid bias toward high-work but low-criticality tasks
    has_slack = (slack > 0.0).astype(float)
    starve_boost_raw = ready_wait_time * upward_rank * has_slack
    starve_denom = np.maximum(remaining_work, eps)
    starve_boost = np.divide(
        starve_boost_raw, starve_denom,
        out=np.full(N, 0.0), where=starve_denom != 0
    )
    starve_boost = np.clip(starve_boost, 0.0, 0.8)

    # Uncertainty-duration penalty only for non-urgent, non-critical-path tasks (to preserve CP integrity)
    cp_mask = (upward_rank >= np.quantile(upward_rank, 0.7, method='lower') if N > 1 else 1.0)
    non_cp_non_urgent = (1.0 - is_urgent) * (1.0 - cp_mask)
    time_uncertainty_penalty = non_cp_non_urgent * uncertainty * task_duration

    # Rank-based normalization (0–1 percentile rank) — stable under low-N, monotonic, bounded
    def percentile_rank(x):
        # Handle ties safely; ensure shape (N,)
        sorted_x = np.sort(x)
        ranks = np.searchsorted(sorted_x, x, side='left') + 1
        return (ranks - 1.0) / max(N - 1, 1)  # maps [min, max] → [0.0, 1.0]

    urgency_rank = percentile_rank(urgency_score)
    energy_rank = percentile_rank(energy_score)
    synergy_rank = percentile_rank(latency_criticality_synergy)
    starve_rank = percentile_rank(starve_boost)
    work_rank = percentile_rank(remaining_work)
    unc_rank = percentile_rank(uncertainty)
    time_penalty_rank = percentile_rank(time_uncertainty_penalty)

    # Final score: lower = better. Prioritize urgency & synergy; penalize energy & uncertainty.
    # Coefficients tuned to enforce deadline-first, then energy-minimization in feasible region.
    score = (
        -3.0 * urgency_rank
        - 2.5 * synergy_rank
        + 0.4 * energy_rank
        + 0.3 * unc_rank
        + 0.25 * time_penalty_rank
        - 0.35 * starve_rank  # negative: higher starvation boost → lower score → higher priority
        + 0.1 * work_rank
    )

    # Final safety: clip, sanitize, ensure shape
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
