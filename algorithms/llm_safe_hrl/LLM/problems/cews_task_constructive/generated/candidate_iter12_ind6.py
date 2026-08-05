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
    Self-evolved priority rule v2: Restores hard deadline adherence via base_urgency,
    fixes starvation inversion, adds slack-sensitivity gating to latency penalty,
    and introduces risk-aware energy reversal with uncertainty-weighted criticality boost.
    
    Key improvements:
      - Restores base_urgency = 1 - sigmoid(slack) as dominant term (0.45) for strict DDL enforcement
      - Fixes starvation_term = 1 - wait_pressure * wait_gate (mitigates starvation, not promotes it)
      - Latency penalty now gated by slack < 0 AND high uncertainty → only penalizes latency when risky & tight
      - Energy reversal enhanced: when slack < 0, favors high-energy-offload *only if* uncertainty is low (safe offload),
        otherwise favors low-energy (conservative fallback)
      - Criticality boost now uncertainty-weighted: amplified_upward_rank * (1 + boost_factor * (1 - normalized_uncertainty))
      - All quantiles use 'higher' method for deterministic small-N stability; robust_minmax_norm handles degeneracy
      - Final score strictly bounded, NaN/inf hardened at every arithmetic stage
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
    median_slack = np.median(slack) + eps
    q1_slack, q3_slack = np.percentile(slack, [25, 75], method='higher')
    iqr_slack = q3_slack - q1_slack + eps

    # Base urgency: 1 - sigmoid(slack) → higher score when slack is negative (urgent), lower when positive (relaxed)
    # Dominant term for hard deadline adherence
    base_urgency = 1.0 / (1.0 + np.exp((slack - median_slack) / (0.25 * iqr_slack + eps)))

    # Slack-gated criticality boost: only amplify upward_rank when slack is tight
    slack_gap = np.clip(median_slack - slack, 0.0, np.inf)
    boost_factor = np.clip(slack_gap / (median_duration + eps), 0.0, 1.0)
    # Uncertainty-aware boost: reduce amplification under high uncertainty (avoid risky critical offloads)
    norm_uncertainty = robust_minmax_norm(uncertainty)
    amplified_upward_rank = upward_rank * (1.0 + boost_factor * (1.0 - norm_uncertainty))

    # Energy reversal: favor low energy when slack > 0 (energy efficiency); 
    # when slack < 0, favor high-energy-offload *only if* uncertainty is low (safe offload), else favor low energy
    norm_energy = robust_minmax_norm(min_incremental_energy)
    energy_reversed = np.where(
        slack > 0,
        norm_energy,  # low energy preferred
        np.where(
            norm_uncertainty < 0.3,
            1.0 - norm_energy,  # high-energy-offload safe under low uncertainty
            norm_energy  # conservative low-energy fallback under high uncertainty
        )
    )

    # Criticality-energy ratio: higher ratio = better critical work per joule → reward it
    crit_energy_ratio = amplified_upward_rank / (min_incremental_energy + eps)
    crit_energy_ratio_clipped = np.clip(crit_energy_ratio, eps, 1e6)
    norm_crit_energy_ratio = robust_minmax_norm(crit_energy_ratio_clipped)

    # Latency penalty: only applied under *both* deadline pressure (slack <= median_slack) AND high uncertainty
    # Prevents unnecessary latency inflation when slack is sufficient or uncertainty is low
    high_uncertainty_mask = (uncertainty >= np.quantile(uncertainty, 0.8, method='higher')).astype(float)
    latency_penalty = np.where(
        (slack <= median_slack) & (high_uncertainty_mask > 0),
        robust_minmax_norm(total_latency) * (1.0 + 0.4 * uncertainty),
        robust_minmax_norm(total_latency)
    )

    # Starvation guard: fixed inversion — now promotes early execution of long-waiting tasks *only when not urgent*
    # (wait_gate activates only when base_urgency < 0.5 → non-critical window)
    wait_threshold = np.quantile(ready_wait_time, 0.9, method='higher') + eps
    wait_pressure = np.clip(ready_wait_time / (wait_threshold + eps), 0.0, 1.0)
    wait_gate = np.where(base_urgency < 0.5, 1.0, 0.0)
    starvation_term = 1.0 - wait_pressure * wait_gate  # smaller score = less starved

    # Work penalty: scaled by slack sensitivity — penalize large remaining_work more when slack is negative
    tau = np.maximum(np.abs(median_slack), 1.0) + eps
    slack_sensitivity = np.exp(-np.clip(np.maximum(-slack, 0.0), 0.0, 100.0) / tau)
    norm_work = robust_minmax_norm(remaining_work)
    work_penalty = norm_work * slack_sensitivity

    # Weighted linear combination — restored urgency dominance, corrected starvation sign, gated latency
    score = (
        0.45 * base_urgency +
        0.25 * energy_reversed +
        0.12 * latency_penalty +
        0.09 * (1.0 - norm_crit_energy_ratio) +
        0.07 * starvation_term +
        0.02 * work_penalty
    )

    # Harden against numerical hazards
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)

    return score
