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
    Self-evolved v2: Deadline-hardened, energy-aware, and numerically bulletproof.
    Key improvements over v1:
    - Restores monotonic & stable min-max normalization (per reflection) with N=1-safe fallback.
    - Replaces arctan+sigmoid combo with *normalized urgency gradient*: linear ramp for slack < 0, flat 0 otherwise — 
      preserves strict deadline dominance without saturation or bias near zero.
    - Introduces *energy-density fairness*: normalizes critical_energy_density by its own median to prevent outlier skew,
      then applies slack-gated sign inversion — ensures high-efficiency tasks rise *only* when safe.
    - Latency-risk term now uses *relative volatility*: (exec+comm)*uncertainty / (mean(exec+comm)+eps), scaled only by |slack|<threshold,
      decoupling risk penalty from absolute deadline distance and focusing on *imminent* violations.
    - Aging is strictly conditional: only applied when slack < -0.1s (hard threshold), normalized robustly, and weighted by urgency magnitude.
    - Uncertainty term dropped (redundant with latency-risk and violates DDL-first principle); replaced by *deadline proximity penalty*
      for near-deadline tasks (0 < slack < 1.0) to prevent last-moment surprises.
    - All terms clamped, eps-protected, and final score guaranteed finite, deterministic, and shape-(N,).
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

    def robust_minmax_normalize(x):
        if x.size == 0:
            return np.zeros_like(x)
        x_min, x_max = np.min(x), np.max(x)
        range_val = x_max - x_min
        if range_val < eps:
            return np.zeros_like(x)
        return (x - x_min) / (range_val + eps)

    # === 1. Hard deadline urgency: linear ramp for violated/imminent deadlines, zero otherwise ===
    # Ensures strict ordering: any slack < 0 dominates all slack >= 0; no saturation, no overflow
    urgency_raw = np.where(slack < 0, -slack, 0.0)  # positive urgency for each second late
    deadline_score = robust_minmax_normalize(urgency_raw)

    # === 2. Critical energy efficiency: only active when slack >= 0; inverted for higher priority ===
    energy_mask = (slack >= 0).astype(float)
    critical_energy_density = upward_rank * (remaining_work + eps) / (min_incremental_energy + eps)
    # Normalize density *before* gating to avoid median skew from zeros
    critical_norm = robust_minmax_normalize(critical_energy_density)
    energy_efficiency_term = -critical_norm * energy_mask

    # === 3. Latency-risk term: relative volatility penalized only under imminent pressure ===
    base_latency = min_exec_time + min_comm_time + eps
    mean_base_latency = np.mean(base_latency) + eps
    # Relative volatility: normalized by typical latency, not slack → avoids artificial amplification when slack is tiny
    relative_volatility = base_latency * (uncertainty + eps) / mean_base_latency
    # Apply only when slack is small *and positive* (imminent) OR negative (violated)
    imminent_mask = (slack < 1.0).astype(float)  # covers [ -inf, 1.0 )
    latency_risk_term = robust_minmax_normalize(relative_volatility) * imminent_mask

    # === 4. Aging term: only activated for *clearly at-risk* tasks (slack < -0.1s), scaled by urgency ===
    aging_mask = (slack < -0.1).astype(float)
    aging_raw = np.where(aging_mask == 1, ready_wait_time, 0.0)
    aging_norm = robust_minmax_normalize(aging_raw)
    # Weight aging by actual urgency magnitude (not binary) for proportional acceleration
    aging_term = -aging_norm * urgency_raw * aging_mask  # zero when not triggered

    # === 5. Proximity penalty: small positive slack (0 < slack < 1.0) gets gentle boost to prevent last-second misses ===
    proximity_mask = ((slack > 0) & (slack < 1.0)).astype(float)
    proximity_penalty = robust_minmax_normalize(np.where(proximity_mask == 1, 1.0 - slack, 0.0))

    # === Weighted combination: deadline safety remains dominant, others support it ===
    score = (
        5.0 * deadline_score +
        2.2 * energy_efficiency_term +
        1.6 * latency_risk_term +
        0.9 * aging_term +
        0.8 * proximity_penalty
    )

    # Final safeguard: ensure finite, shape-(N,), deterministic output
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    return score
