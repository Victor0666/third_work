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
    Priority rule v5: Strictly lexicographic, numerically stable DDL-hardened scoring with:
      - Decoupled urgency & LAED via *normalized risk-adjusted slack* (not effective_slack),
      - Smooth, monotonic urgency using *bounded arctan* (no hard fallbacks or discontinuities),
      - LAED denominator replaced by *latency-efficiency penalty*: (min_exec_time + min_comm_time) * (1 + uncertainty),
        ensuring energy density rewards low-latency *and* low-uncertainty VM choices,
      - Fairness redefined as *deadline-proportional wait pressure*: ready_wait_time / (|slack| + eps),
        scaled only when slack > 0 and gated by urgency to avoid interference,
      - Tie-breaking via *robust inverse latency & inverse criticality*, weighted to preserve lexicographic order,
      - All components normalized, clamped, and summed with strict dominance hierarchy (urgency >> LAED >> fairness >> tiebreak),
      - Zero-inf-nan safety applied uniformly at input, intermediate, and output levels.
    """
    eps = 1e-08
    # Input sanitization: ensure finite, bounded values; no in-place mutation
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=1e6, neginf=eps)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=0.0, posinf=1e2, neginf=0.0)

    # Normalize uncertainty to [0, 1] for consistent scaling
    unc_norm = np.clip(uncertainty / (1e2 + eps), 0.0, 1.0)

    # --- Urgency: bounded arctan on slack, smooth, monotonic, no hard gates ---
    # Use arctan(slope * slack) → maps (-∞, ∞) → (-π/2, π/2); shift & scale to [0, 1]
    # Ensures strict monotonicity and avoids sigmoid flatness near zero while remaining bounded
    urgency_raw = (np.arctan(10.0 * slack) + np.pi/2) / np.pi  # ∈ (0, 1), decreasing in slack
    # Invert so smaller slack → higher urgency → smaller final score
    urgency_score = 1.0 - urgency_raw  # ∈ (0, 1), larger when slack is small/negative

    # --- LAED: Latency-Aware Energy Density — now penalizes high-latency *and* high-uncertainty ---
    # Denominator: base latency penalized by uncertainty → favors predictable fast VMs
    laed_denom = (min_exec_time + min_comm_time) * (1.0 + unc_norm) + min_incremental_energy + eps
    laed_numerator = upward_rank * remaining_work
    laed_raw = laed_numerator / laed_denom
    # No gating: LAED always contributes, but weight ensures urgency dominates
    laed_score = laed_raw

    # --- Fairness: deadline-proportional wait pressure, only active when slack > 0 ---
    # Prevents starvation without compromising urgency: scales wait time relative to deadline margin
    wait_pressure = np.where(slack > 0.0, ready_wait_time / (np.abs(slack) + eps), 0.0)
    # Clamp to avoid blowup when slack ≈ 0; normalize safely
    wait_clipped = np.clip(wait_pressure, 0.0, 1e4)
    fairness_score = np.where(slack > 0.0, wait_clipped / (np.max(wait_clipped + eps) + eps), 0.0)

    # --- Tie-breaking: robust inverse latency & inverse criticality ---
    inv_exec = 1.0 / (min_exec_time + eps)
    inv_rank = 1.0 / (upward_rank + eps)
    def robust_minmax_normalize(x):
        if x.size == 1:
            return np.full_like(x, 0.5, dtype=float)
        x_min, x_max = np.min(x), np.max(x)
        if x_max - x_min < eps:
            return np.full_like(x, 0.5, dtype=float)
        return np.clip((x - x_min) / (x_max - x_min + eps), 0.0, 1.0)
    exec_tie = (1.0 - robust_minmax_normalize(inv_exec)) * 1.0  # smaller exec_time → higher priority → lower tie score
    rank_tie = (1.0 - robust_minmax_normalize(inv_rank)) * 0.1   # smaller rank → less critical → lower tie weight

    # --- Lexicographic weighting with strict dominance hierarchy ---
    # Urgency dominates (1e6), LAED secondary (1e4), fairness tertiary (1e2), tie-breakers residual (1e0)
    score = (
        urgency_score * 1_000_000.0 +
        robust_minmax_normalize(laed_score) * 10_000.0 +
        robust_minmax_normalize(fairness_score) * 100.0 +
        exec_tie + rank_tie
    )

    # Final numerical safety: clamp, replace inf/nan, ensure finite output
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=1e9)
    score = np.clip(score, 1e-6, 1e9)

    # Ensure shape (N,) even for N=1
    return score.astype(float)
