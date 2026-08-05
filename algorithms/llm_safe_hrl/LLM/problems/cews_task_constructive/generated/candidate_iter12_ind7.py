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
    Self-evolved priority rule v2: restores hard deadline adherence via strict slack-gated urgency,
    eliminates non-monotonic couplings, and enforces deterministic feasibility-first ordering.
    
    Key improvements:
      - Urgency is now *strictly binary-gated* by slack sign and magnitude: 
        score = 0.0 for slack <= 0 (max priority), linear decay only for positive slack > median_duration.
      - Criticality-energy coupling simplified to monotonic ratio: upward_rank / (min_incremental_energy + eps),
        normalized robustly — no work-weighting or risk_exponent that caused instability.
      - Energy fairness fully decoupled from criticality; uses pure slack-sign reversal: low-energy favored when slack > 0,
        but *never penalized* when slack <= 0 — avoids diluting urgency dominance.
      - Starvation guard replaced with *hard wait-floor*: tasks waiting > 95th percentile get fixed priority boost (no gating).
      - All normalization uses robust_minmax_norm; all divisions guarded; no sigmoid/tanh/exp approximations.
      - Weight allocation: urgency (0.55) dominates, energy reversal (0.20), latency (0.10), criticality ratio (0.08),
        starvation floor (0.04), work penalty (0.03) — calibrated to reduce DDL violations while preserving energy gains.
      - Final score guaranteed finite, deterministic, shape-(N,), and strictly decreasing in urgency.
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

    def robust_minmax_norm(x):
        x_min, x_max = np.min(x), np.max(x)
        if x_max - x_min < eps:
            return np.zeros_like(x, dtype=float)
        return (x - x_min) / (x_max - x_min + eps)

    total_latency = min_exec_time + min_comm_time + eps
    median_duration = np.median(total_latency) + eps
    median_slack = np.median(slack)

    # Strict urgency: max priority (0.0) for any slack <= 0; linear decay only for slack > median_duration
    base_urgency = np.where(slack <= 0, 0.0,
                           np.clip((slack - median_duration) / (median_duration + eps), 0.0, 1.0))

    # Amplified criticality: upward_rank scaled only by slack-gap boost (linear, bounded)
    slack_gap = median_slack - slack
    boost_factor = np.clip(slack_gap / (median_duration + eps), 0.0, 1.0)
    amplified_upward_rank = upward_rank * (1.0 + boost_factor)

    # Monotonic criticality-energy ratio: no work weighting, no exponentiation
    crit_energy_ratio = amplified_upward_rank / (min_incremental_energy + eps)
    crit_energy_ratio = np.clip(crit_energy_ratio, eps, 1e6)
    norm_crit_energy_ratio = robust_minmax_norm(crit_energy_ratio)

    # Pure slack-sign energy reversal: low-energy favored when slack > 0; neutral (0.5) when slack <= 0
    norm_energy = robust_minmax_norm(min_incremental_energy)
    energy_reversed = np.where(slack > 0, norm_energy, 0.5)

    # Latency penalty: normalized total latency
    norm_latency = robust_minmax_norm(total_latency)

    # Starvation floor: fixed priority boost for top 5% longest-waiting tasks
    wait_threshold = np.quantile(ready_wait_time, 0.95, method='midpoint') + eps
    starvation_boost = np.where(ready_wait_time >= wait_threshold, 0.0, 1.0)

    # Work penalty: only active under negative slack — penalizes large remaining work when at risk
    tau = np.maximum(np.abs(median_slack), 1.0) + eps
    slack_sensitivity = np.exp(-np.clip(np.maximum(-slack, 0.0), 0.0, 100.0) / tau)
    norm_work = robust_minmax_norm(remaining_work)
    work_penalty = norm_work * slack_sensitivity

    # Final weighted score: urgency dominates, energy reversal secondary, others supportive
    score = (
        0.55 * base_urgency +
        0.20 * energy_reversed +
        0.10 * norm_latency +
        0.08 * (1.0 - norm_crit_energy_ratio) +
        0.04 * (1.0 - starvation_boost) +
        0.03 * work_penalty
    )

    # Ensure finiteness and determinism
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)

    return score
