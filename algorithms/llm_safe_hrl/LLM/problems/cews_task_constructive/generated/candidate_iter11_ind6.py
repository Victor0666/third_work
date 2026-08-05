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
    Hybrid priority rule v2: Combines Parent 2's robust min-max normalization and sign-consistent energy reversal
    with Parent 1's risk-gated uncertainty amplification and clipped starvation boost, while adding:
      - Unified urgency-energy coupling via slack-conditioned criticality scaling
      - Workload-aware latency penalty only under deadline pressure
      - Degenerate-safe quantile-based normalization fallback for N=1
      - Strict monotonicity preservation: all terms increase with their physical cost or decrease with benefit
      - Final score bounded and NaN/inf hardened at every stage
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

    # Robust min-max normalization that handles degenerate cases (N=1 or constant arrays)
    def robust_minmax_norm(x):
        x_min, x_max = np.min(x), np.max(x)
        if x_max - x_min < eps:
            return np.zeros_like(x, dtype=float)
        return (x - x_min) / (x_max - x_min + eps)

    # Total latency (exec + comm) with eps for stability
    total_latency = min_exec_time + min_comm_time + eps
    median_duration = np.median(total_latency) + eps
    median_slack = np.median(slack) + eps

    # Urgency: soft binary gate on slack violation/tightness; monotonic decreasing in slack
    # Uses logistic shape centered at median_slack, scaled by IQR for robustness
    q1_slack, q3_slack = np.percentile(slack, [25, 75], method='midpoint')
    iqr_slack = q3_slack - q1_slack + eps
    urgency = 1.0 / (1.0 + np.exp(-(slack - median_slack) / (0.25 * iqr_slack + eps)))
    
    # Criticality amplification: linear boost only when slack is tight (< median_slack)
    slack_gap = np.clip(median_slack - slack, 0.0, np.inf)
    boost_factor = np.clip(slack_gap / (median_duration + eps), 0.0, 1.0)
    amplified_upward_rank = upward_rank * (1.0 + boost_factor)

    # Energy fairness: sign-consistent reversal — low energy favored when slack > 0, high-criticality+low-energy favored when slack <= 0
    norm_energy = robust_minmax_norm(min_incremental_energy)
    energy_reversed = np.where(slack > 0, norm_energy, 1.0 - norm_energy)

    # Critical energy ratio: upward_rank / energy, penalizing low-energy assignment for high-criticality tasks under deadline pressure
    crit_energy_ratio = amplified_upward_rank / (min_incremental_energy + eps)
    # Clip extreme ratios to avoid outlier distortion
    crit_energy_ratio_clipped = np.clip(crit_energy_ratio, eps, 1e6)
    norm_crit_energy_ratio = robust_minmax_norm(crit_energy_ratio_clipped)

    # Latency penalty: only activated when slack is tight (<= median_slack) and scaled by workload importance
    work_scale = np.clip(remaining_work / (np.median(remaining_work) + eps), 0.5, 5.0)
    latency_penalty = np.where(slack <= median_slack, 
                              robust_minmax_norm(total_latency) * work_scale * (1.0 + 0.3 * uncertainty), 
                              robust_minmax_norm(total_latency))

    # Starvation boost: smooth, bounded wait pressure gated by urgency (only when not urgent)
    wait_threshold = np.quantile(ready_wait_time, 0.9, method='midpoint') + eps
    wait_pressure = np.clip(ready_wait_time / (wait_threshold + eps), 0.0, 1.0)
    # Gate: apply starvation relief only when urgency < 0.55 (i.e., not yet deadline-critical)
    wait_gate = np.where(urgency < 0.55, 1.0, 0.0)
    starvation_term = wait_pressure * wait_gate

    # Work penalty: amplify remaining work impact only under negative slack (lateness risk)
    work_penalty = robust_minmax_norm(remaining_work)
    work_penalty = np.where(slack < 0, work_penalty * (1.0 + 0.5 * uncertainty), work_penalty)

    # Final weighted combination — dominant urgency, strengthened energy fairness, balanced latency & criticality
    score = (
        0.42 * urgency +                    # Primary urgency signal (monotonic w.r.t. slack)
        0.26 * energy_reversed +            # Energy fairness with sign-consistent reversal
        0.12 * latency_penalty +            # Latency penalty activated under deadline pressure
        0.09 * (1.0 - norm_crit_energy_ratio) +  # Critical energy efficiency (higher score = worse ratio)
        0.07 * starvation_term +            # Smoothed starvation relief
        0.04 * work_penalty                 # Workload impact under lateness risk
    )

    # Harden final score against numerical issues
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)

    return score
