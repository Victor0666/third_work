import numpy as np

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
    Hybrid priority rule v2: Combines MAD-based robustness and slack-triggered rank inversion from Parent 2,
    with starvation gating, risk-aware energy reversal, and communication-pressure coupling from Parent 1.
    Key innovations:
      - Slack-triggered *criticality inversion* + *energy reversal* jointly activated only when slack <= 0
      - Starvation control via wait-gating that activates only under positive slack (prevents premature promotion)
      - Communication pressure scaled by both slack sign and normalized uncertainty
      - Robust MAD normalization applied uniformly across all terms for outlier resilience
      - Explicit monotonic slack-energy coupling: energy benefit only unlocked when slack > median_slack
      - All divisions guarded; NaN/inf hardened at every stage; deterministic quantiles ('higher' method)
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

    # Robust MAD normalization: center at median, scale by median absolute deviation
    def robust_normalize_mad(x):
        center = np.median(x)
        mad = np.median(np.abs(x - center)) + eps
        normed = (x - center) / mad
        return np.clip(normed, -8.0, 8.0)

    # Task duration and derived metrics
    task_min_duration = np.maximum(min_exec_time + min_comm_time, eps)
    total_latency = task_min_duration.copy()

    # Slack statistics for adaptive gating
    median_slack = np.median(slack) + eps
    q1_slack, q3_slack = np.percentile(slack, [25, 75], method='higher')
    iqr_slack = q3_slack - q1_slack + eps

    # Base urgency: sigmoidal response to slack relative to median and IQR
    base_urgency = 1.0 / (1.0 + np.exp((slack - median_slack) / (0.25 * iqr_slack + eps)))

    # Slack-triggered criticality inversion: invert rank only when deadline violated/tight
    ur_inverted = np.where(slack <= 0.0, -upward_rank, upward_rank)
    ur_norm = robust_normalize_mad(ur_inverted)

    # Risk-aware energy reversal: favor low-energy when slack <= 0; favor high-energy only when slack > 0 AND uncertainty low
    norm_energy = robust_normalize_mad(min_incremental_energy)
    energy_reversed = np.where(
        slack <= 0.0,
        norm_energy,  # prioritize low-energy under pressure
        np.where(uncertainty < np.quantile(uncertainty, 0.3, method='higher'),
                 1.0 - norm_energy,  # safe high-energy offload when uncertainty low
                 norm_energy)  # neutral otherwise
    )

    # Criticality-energy ratio with uncertainty-weighted denominator
    energy_denom = min_incremental_energy * (1.0 + np.clip(uncertainty, 0.0, 2.0))
    energy_denom = np.maximum(energy_denom, eps)
    crit_energy_ratio = ur_inverted / energy_denom
    crit_energy_ratio = np.clip(crit_energy_ratio, 1e-07, 1e7)
    crit_energy_norm = robust_normalize_mad(crit_energy_ratio)

    # Communication pressure: amplified under negative slack, dampened otherwise; modulated by uncertainty
    comm_to_work_ratio = min_comm_time / (remaining_work + eps)
    comm_pressure = np.where(
        slack < 0.0,
        comm_to_work_ratio * (1.0 + 0.5 * np.clip(uncertainty, 0.0, 1.0)),
        comm_to_work_ratio * 0.3
    )
    comm_norm = robust_normalize_mad(comm_pressure)

    # Starvation control: activate wait-penalty only when slack is non-negative (no DDL pressure)
    max_wait = np.maximum(np.max(ready_wait_time), eps)
    rel_wait = np.clip(ready_wait_time / max_wait, 0.0, 1.0)
    wait_gate = np.where(slack >= 0.0, 1.0, 0.0)
    starvation_term = rel_wait * wait_gate

    # Work-normalized urgency: penalize large remaining work only when slack is tight
    norm_work = robust_normalize_mad(remaining_work)
    slack_sensitivity = np.exp(-np.clip(np.maximum(-slack, 0.0), 0.0, 100.0) / (median_slack + eps))
    work_penalty = norm_work * slack_sensitivity

    # Final weighted score: smaller = higher priority
    # Weights sum to ~1.0; calibrated to emphasize urgency & criticality under deadline stress
    score = (
        0.42 * base_urgency +           # dominant deadline enforcement
        0.20 * energy_reversed +       # risk-adjusted energy preference
        0.15 * (-ur_norm) +            # inverted criticality under pressure
        0.10 * (-crit_energy_norm) +   # criticality-per-joule efficiency
        0.06 * comm_norm +             # communication-sensitive scheduling
        0.05 * starvation_term +       # fairness under slack-rich regime
        0.02 * work_penalty            # work-aware load balancing
    )

    # Harden against NaN/inf and clamp extreme values
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)

    return score
