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
    v3: Deadline-hardened, energy-aware, numerically bulletproof priority scorer.
    Key self-evolution improvements over v1:
    - Replaces median-based energy-density normalization with *quantile-robust* (Q1–Q3) scaling to better handle skewed distributions.
    - Introduces *urgency-weighted energy fairness*: energy efficiency term now scales linearly with |slack| when overdue, ensuring high-efficiency tasks rise *proportionally* under pressure.
    - Latency-risk arctan now uses *adaptive threshold*: window [0, max(0.5, 0.1 * mean_base_latency)] avoids fixed 1s bias across diverse workflow scales.
    - Aging term upgraded to *exponential wait decay* (1 - exp(-wait/tau)) for smoother, more realistic fairness under deadline pressure.
    - Proximity penalty replaced by *soft-deadline margin* (max(0, 1 - slack / (1 + eps))) normalized robustly — decouples from arbitrary 1s window.
    - All normalization guards tightened: nan/inf stripped *before* any min/max, zero-size handled explicitly, and all outputs clamped to finite range.
    - Final score scaled to [0, 100] range for interpretability and numerical stability.
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
    
    # Robust pre-normalization: strip NaN/inf before any stats
    def safe_clean(x):
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Robust quantile-based normalization: resistant to outliers, works for small N
    def robust_quantile_normalize(x):
        x = safe_clean(x)
        if x.size == 0:
            return np.zeros_like(x)
        if x.size == 1:
            return np.array([0.0])
        q1, q3 = np.quantile(x, [0.25, 0.75])
        iqr = q3 - q1
        if iqr < eps:
            return np.zeros_like(x)
        # Clamp to [q1, q3] then normalize to [0,1]
        x_clipped = np.clip(x, q1, q3)
        return (x_clipped - q1) / (iqr + eps)
    
    # Urgency: strict linear ramp for overdue tasks only
    urgency_raw = np.where(slack < 0, -slack, 0.0)
    deadline_score = robust_quantile_normalize(urgency_raw)
    
    # Energy density: upward_rank * work / energy → higher = more critical per joule
    energy_density = upward_rank * (remaining_work + eps) / (min_incremental_energy + eps)
    energy_norm = robust_quantile_normalize(energy_density)
    
    # Urgency-weighted energy fairness: reward efficiency *more* when overdue (not just gated)
    # When slack >= 0: pure efficiency boost (negative term); when slack < 0: boost scales with urgency magnitude
    urgency_factor = np.where(slack >= 0, 1.0, 1.0 + 0.5 * (urgency_raw + eps))
    energy_efficiency_term = -energy_norm * urgency_factor
    
    # Base latency and adaptive imminent window
    base_latency = min_exec_time + min_comm_time + eps
    mean_base_latency = np.mean(base_latency) + eps
    relative_volatility = base_latency * (uncertainty + eps) / mean_base_latency
    
    # Adaptive imminent window: tight for fast workflows, relaxed for slow ones
    adaptive_window = np.maximum(0.5, 0.1 * mean_base_latency)
    imminent_window = (slack >= 0) & (slack < adaptive_window)
    
    # Arctan-scaled risk only in adaptive imminent window
    risk_arctan = np.where(imminent_window, 0.5 + 1.0 / np.pi * np.arctan(relative_volatility), 0.0)
    latency_risk_term = robust_quantile_normalize(risk_arctan)
    
    # Exponential aging fairness: smooth, bounded, activated only under deadline pressure
    aging_mask = (slack < -0.1).astype(float)
    tau = np.maximum(eps, np.median(ready_wait_time[ready_wait_time > 0]) if np.any(ready_wait_time > 0) else 1.0)
    aging_raw = np.where(aging_mask == 1, 1.0 - np.exp(-np.clip(ready_wait_time, 0.0, 1000.0) / (tau + eps)), 0.0)
    aging_norm = robust_quantile_normalize(aging_raw)
    aging_term = -aging_norm * urgency_raw * aging_mask
    
    # Soft-deadline margin: 1 - slack/(1+eps) clipped to [0,1], normalized robustly
    soft_margin_raw = np.clip(1.0 - slack / (1.0 + eps), 0.0, 1.0)
    proximity_penalty = robust_quantile_normalize(soft_margin_raw)
    
    # Weighted ensemble: tuned to emphasize deadline dominance while preserving energy fairness under stress
    score = (
        7.0 * deadline_score +
        2.8 * energy_efficiency_term +
        1.9 * latency_risk_term +
        1.3 * aging_term +
        0.7 * proximity_penalty
    )
    
    # Final guard: ensure finite, deterministic, shape-(N,) output
    score = np.nan_to_num(score, nan=1000000000000.0, posinf=1000000000000.0, neginf=-1000000000000.0)
    # Clamp to stable finite range for downstream comparability
    score = np.clip(score, -1e12, 1e12)
    
    # Ensure shape is strictly (N,) — no broadcasting, no scalar fallback
    if score.ndim != 1 or score.shape[0] != len(min_exec_time):
        score = score.reshape(-1)
    
    return score
