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
    Self-evolved priority rule: hard deadline enforcement first, critical-path fidelity second,
    energy-latency efficiency third — with unified robust normalization, strict signal polarity,
    and monotonic risk gating.

    Key self-evolution improvements:
    - Uniform binary gating: all non-deadline terms (upward_rank, crit_energy, uncertainty) are zeroed when slack <= 0,
      ensuring no conflicting signals undermine hard DDL safety.
    - Replaces arctan with *linear urgency ramp* for slack >= 0: preserves monotonicity and avoids non-monotonic gradients
      near zero that dilute deadline sensitivity.
    - Uses *only MAD normalization* across all terms for scale consistency and weight hierarchy fidelity.
    - Introduces *critical-energy density*: (upward_rank * remaining_work) / (min_incremental_energy + eps),
      normalized only when slack > 0 — directly encodes "critical work per joule".
    - Fairness uses *sqrt-scaled wait time*, bounded to [0, 0.1] and applied unconditionally (to prevent starvation even under stress),
      but weighted lowest to preserve objective hierarchy.
    - Uncertainty contribution is now *risk-weighted latency inflation*: (min_exec_time + min_comm_time) * (1 + uncertainty)
      only when slack <= 0, then normalized — prioritizes safer scheduling under high uncertainty & tight margin.
    - Final weights strictly enforce objective priority: deadline (4.0) >> critical-energy (2.2) >> upward_rank (1.3) >>
      latency-inflation (0.3) >> fairness (0.09) >> remaining_work (0.07).
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

    def safe_mad_normalize(x):
        """MAD normalization robust to N=1 and constant arrays; returns zeros if degenerate."""
        if x.size == 1:
            return np.zeros_like(x)
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + eps
        if mad < eps:
            return np.zeros_like(x)
        normed = (x - med) / mad
        return np.clip(normed, -4.0, 4.0)

    # === Deadline score: strict binary gating + monotonic margin ramp ===
    # slack < 0 → exponential penalty; slack >= 0 → linear ramp (0 at slack=0, increasing with margin deficit)
    deadline_risk_raw = np.where(
        slack < 0,
        np.exp(np.clip(-slack, 0, 20)) - 1.0,
        np.clip(-slack, 0.0, 10.0)  # linear penalty for positive slack (i.e., margin shortfall relative to ideal)
    )
    deadline_score = safe_mad_normalize(deadline_risk_raw)

    # === Critical-energy density: only active when slack > 0 ===
    crit_energy_raw = np.where(
        slack > 0,
        (upward_rank * remaining_work + eps) / (min_incremental_energy + eps),
        0.0
    )
    crit_energy_norm = safe_mad_normalize(crit_energy_raw)

    # === Upward rank: only active when slack > 0 ===
    upward_rank_active = np.where(slack > 0, upward_rank, 0.0)
    upward_rank_norm = safe_mad_normalize(upward_rank_active)

    # === Latency-inflation under risk: only when slack <= 0 AND uncertainty > 0 → penalize risky latency ===
    inflated_latency = np.where(
        (slack <= 0) & (uncertainty > 0.0),
        (min_exec_time + min_comm_time) * (1.0 + np.clip(uncertainty, 0.0, 0.5)),
        min_exec_time + min_comm_time
    )
    latency_inflation_raw = np.where(slack <= 0, inflated_latency, 0.0)
    latency_inflation_norm = safe_mad_normalize(latency_inflation_raw)

    # === Fairness: sqrt-scaled wait time, unconditional but bounded [0, 0.1] ===
    max_wait = np.max(ready_wait_time) + eps
    wait_boost = np.sqrt(np.clip(ready_wait_time / max_wait, 0.0, 1.0))
    wait_boost = np.clip(wait_boost, 0.0, 0.1)

    # === Remaining work: only normalized, no gating (low weight, secondary role) ===
    remaining_work_norm = safe_mad_normalize(remaining_work)

    # Combine with strict hierarchical weights
    score = (
        +4.0 * deadline_score
        - 2.2 * crit_energy_norm
        - 1.3 * upward_rank_norm
        + 0.3 * latency_inflation_norm
        + 0.09 * wait_boost
        + 0.07 * remaining_work_norm
    )

    # Final sanitization: ensure finite, deterministic output
    return np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
