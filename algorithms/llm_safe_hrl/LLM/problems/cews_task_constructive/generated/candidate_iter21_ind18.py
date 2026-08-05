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
    v2 self-evolved priority rule: Deadline-hardened, energy-efficient, and signal-discriminative.
    Key evolutions from v1:
      - Replaces global MAD normalization with *adaptive per-signal robust scaling*:
        each base term is scaled by its own IQR-based spread (not shared MAD), preserving
        inter-signal dynamic range while avoiding over-normalization damping.
      - Introduces *deadline violation margin penalty*: instead of hard offset (-1e9),
        uses smooth, differentiable penalty: -1e6 * sigmoid(-slack/0.1) — penalizes
        imminent violations strongly but gracefully, improving gradient alignment.
      - Refines SCD: multiplies by `remaining_work` to weight criticality by total downstream load,
        yielding Slack-Constrained Critical Load Density (SCLD) = upward_rank * remaining_work * sigmoid(slack/τ) / (exec+comm).
      - Adds *energy-latency tradeoff term*: min_incremental_energy / (min_exec_time + min_comm_time + ε),
        normalized and negatively weighted — prioritizes low-energy-per-latency tasks when safe.
      - Fairness now uses *log(1+wait)* instead of sqrt(wait) for better small-wait discrimination,
        still gated by slack > 0 and capped at 0.2.
      - All terms are zero/Nan/inf protected; deterministic; returns shape-(N,).
      - Final coefficients prioritize deadline safety > energy-latency efficiency > critical load > fairness.
    """
    eps = 1e-08
    N = len(np.atleast_1d(slack))
    if N == 0:
        return np.array([], dtype=float)

    # Defensive casting and NaN/inf handling
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=eps, neginf=eps)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=eps, posinf=eps, neginf=eps)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    base_signals = []

    # 1. Urgency: tanh-based, clipped for stability
    tau_urg = 0.5
    urgency_raw = np.tanh(-np.clip(slack, -100.0, 100.0) / tau_urg)
    base_signals.append(urgency_raw)

    # 2. SCLD: Slack-Constrained Critical Load Density — upward_rank * remaining_work gated by slack
    slack_gate_scd = 1.0 / (1.0 + np.exp(-slack / (tau_urg + eps)))
    exec_comm_sum = min_exec_time + min_comm_time + eps
    scl_d_raw = upward_rank * remaining_work * slack_gate_scd / exec_comm_sum
    base_signals.append(scl_d_raw)

    # 3. Energy-efficiency gated by slack (same as v1, retained for proven energy awareness)
    energy_efficiency = 1.0 / (min_incremental_energy + eps)
    slack_gate_energy = 1.0 / (1.0 + np.exp(-slack / 1.0))
    seer_gated = energy_efficiency * slack_gate_energy
    base_signals.append(seer_gated)

    # 4. Energy-latency tradeoff: lower energy per latency unit preferred when slack permits
    energy_per_latency = min_incremental_energy / (exec_comm_sum + eps)
    # Gate it: only activate when slack is *sufficient* (>= 1s), else suppress
    tradeoff_gate = np.where(slack >= 1.0, 1.0, np.exp(slack - 1.0))  # smooth ramp-down
    tradeoff_raw = energy_per_latency * tradeoff_gate
    base_signals.append(tradeoff_raw)

    # 5. Fairness: log(1+wait) improves resolution for small wait times, still slack-gated
    wait_safe = np.maximum(ready_wait_time, 0.0)
    log_wait = np.log1p(wait_safe)  # log(1+x), avoids log(0)
    fairness_raw = np.where(slack > 0.0, log_wait * np.exp(-uncertainty), 0.0)
    fairness_clipped = np.clip(fairness_raw, 0.0, 0.2)
    base_signals.append(fairness_clipped)

    # 6. Smooth deadline violation penalty (replaces hard offset) — differentiable & calibrated
    violation_penalty = -1e6 * (1.0 / (1.0 + np.exp(slack / 0.1)))  # sig(-slack/0.1) ≈ 1 when slack << 0
    base_signals.append(violation_penalty)

    # Adaptive per-signal robust scaling using IQR (interquartile range), not shared MAD
    # Ensures each term contributes discriminatively without cross-term suppression
    def robust_scale(x):
        q25, q75 = np.percentile(x, [25, 75])
        iqr = q75 - q25
        center = np.median(x)
        scale = iqr if iqr > eps else (np.max(x) - np.min(x) + eps)
        return (x - center) / (scale + eps)

    norm_urgency = robust_scale(urgency_raw)
    norm_scl_d = robust_scale(scl_d_raw)
    norm_seer = robust_scale(seer_gated)
    norm_tradeoff = robust_scale(tradeoff_raw)
    norm_fairness = robust_scale(fairness_clipped)
    norm_violation = robust_scale(violation_penalty)

    # Weighted linear combination: strict deadline priority first, then efficiency & load
    # Coefficients reflect objective hierarchy: DDL-hard → energy-latency → critical load → fairness
    urgency_term = -6.0 * norm_urgency
    scl_d_term = -2.2 * norm_scl_d
    seer_term = -1.5 * norm_seer
    tradeoff_term = -3.0 * norm_tradeoff  # higher weight: energy-per-latency is key efficiency proxy
    fairness_term = -0.5 * norm_fairness
    violation_term = -1.0 * norm_violation  # scaled to avoid overwhelming other terms

    score = (urgency_term + scl_d_term + seer_term + tradeoff_term +
             fairness_term + violation_term)

    # Final safeguard: finite, bounded, shape-(N,)
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)
    return score.reshape(-1)
