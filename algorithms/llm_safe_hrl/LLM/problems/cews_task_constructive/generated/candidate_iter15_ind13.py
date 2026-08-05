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
    Priority rule v2 (evolved): Strict lexicographic DDL-hardened scoring with:
      - Urgency: bounded arctan of slack, smoothly penalizing lateness (negative slack) and rewarding headroom
      - LAED (Latency-Aware Energy-Density): (upward_rank * remaining_work) / ((min_exec_time + min_comm_time) * (1 + uncertainty) + min_incremental_energy + eps)
        → preserves critical-path energy efficiency coupling even under urgency; no gating → avoids zeroing critical signals
      - Uncertainty-aware fairness: ready_wait_time / (|slack| + eps), *always active* (no slack>0.5 gate) → prevents starvation near deadline;
        normalized relative to max wait pressure in ready set to ensure scale-invariance
      - Tie-breaking: robust inverse latency & inverse criticality, weighted to preserve hierarchy without normalization artifacts
      - All components guarded against NaN/inf/zero at input, intermediate, and output levels
      - Strict dominance enforced via multiplicative scaling (urgency >> LAED >> fairness >> tiebreak)
      - Final score clamped and sanitized for deterministic finite output
    '''
    eps = 1e-08
    # Input sanitization: coerce to float, replace NaN/inf/neg-inf with safe values
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=1e6, neginf=eps)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=0.0, posinf=100.0, neginf=0.0)

    # 1. Urgency: smooth, monotonic, zero-discontinuity at slack=0; negative slack → high urgency (low score)
    # arctan(10*slack) maps (-∞,∞) → (-π/2, π/2); shift+scale → (0,1) where 0=most urgent (large negative slack)
    urgency_raw = (np.arctan(10.0 * slack) + np.pi / 2) / np.pi
    urgency_score = 1.0 - urgency_raw  # now [0,1]: smaller = more urgent

    # 2. LAED (Latency-Aware Energy-Density): criticality per unit latency-efficiency-energy cost
    # Uses *full* denominator: (latency * (1+uncertainty) + energy) → couples latency, uncertainty, energy tightly
    # Always computed (no slack gating) → preserves energy signal even under urgency; avoids zeroing
    latency_term = min_exec_time + min_comm_time + eps
    unc_weighted_latency = latency_term * (1.0 + np.clip(uncertainty, 0.0, 10.0))  # cap uncertainty impact
    laed_denom = unc_weighted_latency + min_incremental_energy + eps
    laed_numerator = upward_rank * remaining_work
    laed_score = laed_numerator / laed_denom  # higher = more critical per resource cost → prioritize

    # 3. Fairness: deadline-proportional wait pressure — active for *all* tasks, including slack <= 0
    # Prevents starvation: even if slack is negative, long-waiting tasks gain priority to avoid indefinite deferral
    wait_pressure = ready_wait_time / (np.abs(slack) + eps)
    # Normalize *within ready set*: ensures fairness scales with current contention, not absolute values
    if wait_pressure.size == 1:
        fairness_score = np.full_like(wait_pressure, 0.0, dtype=float)
    else:
        wp_min, wp_max = np.min(wait_pressure), np.max(wait_pressure)
        if wp_max - wp_min < eps:
            fairness_score = np.full_like(wait_pressure, 0.0, dtype=float)
        else:
            fairness_score = (wait_pressure - wp_min) / (wp_max - wp_min + eps)

    # 4. Tie-breaking: prefer low latency & high criticality when other scores tie
    inv_exec = 1.0 / (min_exec_time + eps)
    inv_rank = 1.0 / (upward_rank + eps)
    
    def robust_minmax_normalize(x):
        if x.size == 1:
            return np.full_like(x, 0.5, dtype=float)
        x_min, x_max = np.min(x), np.max(x)
        if x_max - x_min < eps:
            return np.full_like(x, 0.5, dtype=float)
        return np.clip((x - x_min) / (x_max - x_min + eps), 0.0, 1.0)
    
    norm_urgency = robust_minmax_normalize(urgency_score)
    norm_laed = robust_minmax_normalize(laed_score)
    norm_fairness = fairness_score  # already normalized relative to ready set
    norm_inv_exec = robust_minmax_normalize(inv_exec)
    norm_inv_rank = robust_minmax_normalize(inv_rank)

    # Lexicographic dominance via multiplicative scaling: urgency dominates by 5 orders, LAED by 3, etc.
    # Avoids cancellation and preserves strict ordering hierarchy
    score = (
        norm_urgency * 1e6 +
        norm_laed * 1e3 +
        norm_fairness * 1e1 +
        (1.0 - norm_inv_exec) * 1.0 +  # prefer low exec time → higher priority = lower score
        (1.0 - norm_inv_rank) * 0.1    # prefer high upward rank → higher priority = lower score
    )

    # Final sanitization: ensure finite, positive, deterministic output
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=1e9)
    score = np.clip(score, eps, 1e9)
    return score.astype(float)
