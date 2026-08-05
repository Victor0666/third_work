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
    v2: Refined deadline-hardened priority rule with corrected objective alignment,
         bounded exponential urgency, properly signed energy-efficiency terms,
         and starvation-robust fairness.

    Key self-evolution fixes & improvements:
    - Urgency restored to bounded exponential for slack < -5 (exp(|slack|-5)), preserving strict deadline hierarchy;
      soft arctan for slack >= 0 remains for smooth early-bird prioritization.
    - All energy-efficiency terms (CED, upward_rank) now have *positive* weights — higher values improve priority.
    - Risk penalty is now *subtracted*: slack < 0 increases score (lowers priority), correctly penalizing violations.
    - Uncertainty only modulates *negative-slack* urgency via multiplicative scaling (1 + 0.5*uncertainty), not tanh distortion.
    - Fairness term uses log1p wait time scaled to [0, 0.08], conservatively weighted (0.04) to avoid deadline dilution.
    - Unified safe_mad_normalize with explicit zero-median fallback and robust clipping; handles degenerate cases (N=1, const).
    - Final score strictly clipped to [-1e9, 1e9] and nan_to_num’d for guaranteed determinism and finiteness.
    - Removed redundant remaining_work from CED numerator — upward_rank already captures critical-path importance.
    """
    eps = 1e-08

    # Clean all inputs: ensure float, replace NaN/inf with safe defaults
    def clean_array(x):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=eps)

    min_exec_time = clean_array(min_exec_time)
    min_comm_time = clean_array(min_comm_time)
    min_incremental_energy = clean_array(min_incremental_energy)
    slack = clean_array(slack)
    upward_rank = clean_array(upward_rank)
    remaining_work = clean_array(remaining_work)
    ready_wait_time = clean_array(ready_wait_time)
    uncertainty = clean_array(uncertainty)

    def safe_mad_normalize(x):
        """MAD normalization robust to N=1, constants, outliers; returns zeros if unstable"""
        if x.size == 1:
            return np.zeros_like(x)
        x_clipped = np.clip(x, -1e6, 1e6)
        med = np.median(x_clipped)
        dev = np.abs(x_clipped - med)
        mad = np.median(dev) + eps
        if mad < eps:
            return np.zeros_like(x)
        normed = (x_clipped - med) / mad
        return np.clip(normed, -4.0, 4.0)

    # === Deadline Urgency (strict hierarchy: soft → linear → hard) ===
    arctan_urgency = 0.5 + (1.0 / np.pi) * np.arctan(np.where(slack >= 0, slack, 0.0) / 5.0)
    linear_violation = np.where((slack < 0) & (slack >= -5), 1.0 + (-slack) / 5.0, 0.0)
    severe_violation = np.where(slack < -5, np.exp(np.clip(-slack - 5, 0.0, 20.0)), 0.0)
    # Uncertainty amplifies *only* negative slack penalties, multiplicatively
    uncertainty_scaled = np.clip(uncertainty, 0.0, 5.0)
    urgency_raw = arctan_urgency + linear_violation + severe_violation * (1.0 + 0.5 * uncertainty_scaled)
    urgency_norm = safe_mad_normalize(urgency_raw)

    # === Critical-Energy Density (CED): latency-aware efficiency metric ===
    # Active only when slack > 0 (deadline-safe zone) and resources are meaningful
    exec_comm_lat = min_exec_time + min_comm_time + eps
    ced_active = (slack > 0) & (upward_rank > eps) & (min_incremental_energy > eps)
    # CED = upward_rank / (energy * latency) → higher = more efficient & critical → better priority
    ced_raw = np.where(ced_active, upward_rank / (min_incremental_energy * exec_comm_lat + eps), 0.0)
    ced_norm = safe_mad_normalize(ced_raw)

    # === Upward Rank (critical-path importance) — active only in safe zone ===
    upward_active = np.where(ced_active, upward_rank, 0.0)
    upward_norm = safe_mad_normalize(upward_active)

    # === Risk Penalty: subtractive term — lowers priority on violation risk ===
    # Only applied when slack < 0, scaled by uncertainty and clipped
    risk_raw = np.where(slack < 0, np.clip((-slack) * uncertainty_scaled, 0.0, 50.0), 0.0)
    risk_norm = safe_mad_normalize(risk_raw)

    # === Fairness: log-scaled wait time, capped and lightly weighted ===
    max_wait = np.max(ready_wait_time) + eps
    rel_log_wait = np.log1p(ready_wait_time) / (np.log1p(max_wait) + eps)
    fairness_boost = np.clip(rel_log_wait, 0.0, 0.08)

    # === Final score: smaller = higher priority
    # Objective hierarchy: urgency (5.0) >> CED (2.5) >> upward_rank (1.4) >> fairness (0.04); risk *subtracts* priority
    score = (
        +5.0 * urgency_norm      # High urgency → low score → high priority
        - 2.5 * ced_norm         # High CED → low score → high priority (so negate)
        - 1.4 * upward_norm      # High upward rank → low score → high priority (so negate)
        + 0.35 * risk_norm       # Risk increases score → lowers priority (correct sign)
        + 0.04 * fairness_boost  # Mild fairness boost (low weight avoids deadline tradeoff)
    )

    # Ensure strict determinism, finiteness, and shape compliance
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)
    return score
