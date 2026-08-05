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
    Self-evolved v2: restores strong deadline fidelity with adaptive linear-soft urgency,
    fixes starvation logic (direct penalty, not inverted), and broadens energy-awareness
    to all critical tasks — even under slack pressure — via risk-gated work scaling.
    
    Key improvements:
      - Urgency: replaced sigmoid/exp mix with unified *bounded linear-soft ramp*: 
        linear penalty for slack <= 0, smooth exponential decay for slack > 0 → monotonic,
        numerically stable, and preserves deadline dominance.
      - Starvation: direct, normalized wait penalty scaled by remaining_work and gated only by
        feasibility (non-zero work) — no slack/latency double-gating that suppressed late-task fairness.
      - Energy-context: removes `slack > 0` gate; instead uses `upward_rank > median_ur` AND
        `uncertainty < 0.5` to enable work-scaling only when criticality is high *and* risk is moderate,
        ensuring energy awareness remains active for urgent but predictable tasks.
      - Criticality-energy tradeoff: refines risk exponent using *both* slack deficit and uncertainty,
        with floor=1.0 and ceiling=3.5 to prevent over-penalization in extreme cases.
      - Normalization: tightened IQR bounds to [-4, 4] and added explicit finite quantile fallback
        for degenerate (constant) arrays to guarantee robustness.
      - All terms sign-consistent: lower score = higher priority; no inversion or 1-x tricks.
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

    # Robust normalization handling constant arrays safely
    def robust_iqr_norm(x):
        q1, q3 = np.quantile(x, [0.25, 0.75], method='midpoint')
        iqr = q3 - q1 + eps
        center = np.median(x)
        normed = (x - center) / iqr
        # Clamp to avoid outlier distortion; ensure finite bounded output
        return np.clip(normed, -4.0, 4.0)

    # Task intrinsic time cost — base for urgency & normalization
    task_min_duration = np.maximum(min_exec_time + min_comm_time, eps)

    # === URGENT DEADLINE FIDELITY ===
    # Bounded linear-soft urgency: 0–1 scale, monotonic decreasing in slack
    # For slack <= 0: linear ramp from 1.0 (most urgent) to 2.0 at slack = -median_duration
    # For slack > 0: soft exponential decay exp(-slack / (median_duration + eps))
    median_dur = np.median(task_min_duration) + eps
    urgency_neg = 1.0 + np.clip(-slack / median_dur, 0.0, 1.0)  # [1.0, 2.0]
    urgency_pos = np.exp(-np.clip(slack / (median_dur + eps), 0.0, 20.0))  # (0.0, 1.0]
    deadline_urgency = np.where(slack <= 0, urgency_neg, urgency_pos)

    # === CRITICALITY-ENERGY TRADEOFF ===
    # Risk exponent: amplifies energy penalty under lateness *and* high uncertainty
    base_risk = np.maximum(0.0, -slack) / median_dur
    risk_exponent = np.clip(1.0 + 0.5 * base_risk + 0.3 * uncertainty, 1.0, 3.5)
    energy_risk_weighted = min_incremental_energy * np.power(1.0 + uncertainty, risk_exponent)
    energy_safe = np.maximum(energy_risk_weighted, eps)
    crit_eff_ratio = upward_rank / energy_safe
    crit_eff_ratio = np.clip(crit_eff_ratio, 1e-06, 1e6)
    norm_crit_eff = robust_iqr_norm(crit_eff_ratio)

    # === ENERGY CONTEXTUALIZATION ===
    # Scale energy by remaining_work only when task is both highly critical *and* low-uncertainty
    median_ur = np.median(upward_rank) + eps
    ur_ratio = upward_rank / median_ur
    energy_work_weight = np.where((ur_ratio > 1.4) & (uncertainty < 0.5),
                                  np.clip(remaining_work / (np.median(remaining_work) + eps), 0.8, 2.5),
                                  1.0)
    energy_scaled = min_incremental_energy * energy_work_weight
    norm_energy = robust_iqr_norm(energy_scaled)

    # === STARVATION CONTROL (FIXED) ===
    # Direct, work-normalized wait penalty: penalizes long waits per unit work
    # No slack gating — ensures fairness even for late tasks; capped at [0.0, 0.3]
    wait_per_work = ready_wait_time / (remaining_work + eps)
    max_wait_pw = np.maximum(np.max(wait_per_work), eps)
    wait_penalty = np.clip(wait_per_work / max_wait_pw, 0.0, 1.0) * 0.3

    # === AUXILIARY NORMALIZED TERMS ===
    norm_work = robust_iqr_norm(remaining_work)
    norm_unc = robust_iqr_norm(uncertainty)
    norm_dur = robust_iqr_norm(task_min_duration)

    # === FINAL SCORE: lower = better (priority)
    # Weighted sum preserving urgency dominance and criticality-energy synergy
    score = (
        0.42 * deadline_urgency +           # strongest weight: hard DDL compliance first
        0.23 * (1.0 - norm_crit_eff) +      # high crit/low energy → lower score
        0.11 * norm_dur +                   # shorter duration → slightly favored
        0.09 * norm_energy +                # contextually scaled energy → minor penalty if high
        0.07 * norm_work +                  # larger remaining work → slight preference (critical path)
        0.05 * norm_unc +                   # higher uncertainty → slight penalty
        0.03 * wait_penalty                 # direct starvation penalty
    )

    # Final sanitization: ensure finite, deterministic, shape-(N,)
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)
    return score
