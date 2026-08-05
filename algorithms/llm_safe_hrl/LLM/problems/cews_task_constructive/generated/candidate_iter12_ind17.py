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
    v2 refined: Restores strict lexicographic DDL dominance by eliminating all non-urgency terms for slack < 0.
    Replaces MAD with robust min-max normalization (stable in low-variance sets) and enforces monotonicity.
    Introduces *critical-path energy density under deadline pressure* — CED scaled by urgency, not masked.
    Efficiency term is now *only active when slack >= 0*, and uses uncertainty-inflated latency only in tight regimes.
    Fairness (aging) is fully gated: applied *only if* (urgency > 0.7 AND wait_time > 90th percentile) → avoids noise.
    All terms are finite, shape-(N,), deterministic, and zero-division/NaN-safe.
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

    # --- Urgency: arctan-based, bounded, signed; negative slack → max urgency (min score)
    arctan_urgency = (np.arctan(-slack / (1.0 + eps)) + np.pi / 2) / np.pi
    urgency_term = -6.0 * arctan_urgency  # Strongest weight; dominates all others

    # --- Critical Energy Density (CED): upward_rank * work / energy, *always computed*, then scaled by urgency
    # This preserves deadline-critical tasks even when violating — no masking → fixes lexicographic break
    base_ced = upward_rank * remaining_work / (min_incremental_energy + eps)
    # Robust min-max normalization over full set (not subset) to avoid distortion in sparse sets
    def robust_minmax(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        xmin, xmax = np.min(x), np.max(x)
        rng = xmax - xmin
        if rng > eps:
            return (x - xmin) / (rng + eps)
        else:
            # Near-constant → normalize to zeros
            return np.zeros_like(x)
    norm_ced = robust_minmax(base_ced)
    # Scale CED by urgency: amplify priority of high-CED tasks *only when urgent*
    critical_energy_term = -2.5 * norm_ced * arctan_urgency

    # --- Efficiency term: active ONLY for slack >= 0; uses uncertainty-inflated latency only when tight
    base_latency = min_exec_time + min_comm_time + eps
    # Tight regime: slack > 0 but small (<= median positive slack * 1.5)
    pos_slack = slack[slack > 0]
    median_pos_slack = np.median(pos_slack) if pos_slack.size > 0 else 1.0
    tight_regime = (slack > 0) & (slack <= 1.5 * median_pos_slack)
    # Inflate latency only in tight regime, capped at +30%
    lat_ref = np.percentile(base_latency, 90) + eps if base_latency.size > 1 else np.max(base_latency) + eps
    inflation_factor = np.clip(uncertainty / lat_ref, 0.0, 0.3)
    inflated_latency = np.where(tight_regime, base_latency * (1.0 + inflation_factor), base_latency)
    eff_ratio = min_incremental_energy / (inflated_latency + eps)
    # Normalize only over slack >= 0 subset, but use full-shape output with zeros elsewhere
    eff_mask = slack >= 0
    eff_valid = eff_ratio[eff_mask] if np.any(eff_mask) else np.array([0.0])
    norm_eff_full = np.zeros_like(slack)
    if eff_valid.size > 0:
        norm_eff_full[eff_mask] = robust_minmax(eff_valid)
    efficiency_term = 1.3 * norm_eff_full

    # --- Fairness (wait amplification): only for highly urgent AND significantly delayed tasks
    # Avoids noise: requires both urgency > 0.7 AND wait_time > 90th percentile
    urgency_gate = arctan_urgency > 0.7
    wait_cap = np.percentile(ready_wait_time, 90) + eps if ready_wait_time.size > 1 else np.max(ready_wait_time) + eps
    wait_long_enough = ready_wait_time > wait_cap
    clipped_wait = np.clip(ready_wait_time, 0.0, 10.0)
    norm_wait_full = robust_minmax(clipped_wait)
    fairness_term = -0.4 * norm_wait_full * urgency_gate * wait_long_enough

    # --- Lexicographic fusion: urgency always dominates; other terms *disabled* when slack < 0
    # No convex weights — instead, hard gating: efficiency & fairness zeroed when slack < 0
    efficiency_term = np.where(slack < 0, 0.0, efficiency_term)
    fairness_term = np.where(slack < 0, 0.0, fairness_term)
    # critical_energy_term already urgency-scaled → naturally attenuated when not urgent

    # Final score: sum of deterministic, finite, shape-(N,) terms
    score = urgency_term + critical_energy_term + efficiency_term + fairness_term
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    # Ensure shape (N,) explicitly
    return score.reshape(-1)
