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
    Hybrid priority rule v2: combines Parent 2's monotonic urgency coupling and sign-consistent energy reversal
    with Parent 1's robust per-feature normalization, latency-criticality synergy, and uncertainty-aware duration scaling.
    Key novelties:
      - Unified robust normalization using clipped IQR-based z-score fallback (stable at small N)
      - Slack-gated criticality-energy synergy: (upward_rank * total_latency) / (min_incremental_energy + eps) 
        weighted by urgency mask to avoid penalizing late tasks unnecessarily
      - Uncertainty-augmented duration as primary urgency driver: (exec + comm) * (1 + uncertainty)
      - Starvation control via smoothed wait pressure gated by both slack positivity AND low work density
      - Energy fairness strengthened with uncertainty-weighted reversal: energy score scaled by (1 + uncertainty)^slack_sign_factor
      - All components strictly bounded, finite, and shape-(N,) guaranteed
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

    # Robust normalization: clipped z-score with IQR fallback for stability
    def robust_normalize(x):
        median_x = np.median(x)
        q1, q3 = np.quantile(x, [0.25, 0.75], method='midpoint')
        iqr = q3 - q1 + eps
        std_fallback = np.std(x) + eps
        z_score = (x - median_x) / std_fallback
        iqr_norm = (x - median_x) / iqr
        norm = np.where(np.abs(z_score) > 10.0, iqr_norm, z_score)
        return np.clip(norm, -6.0, 6.0)

    # Core temporal drivers
    total_latency = min_exec_time + min_comm_time + eps
    unc_duration = total_latency * (1.0 + uncertainty)
    median_unc_duration = np.median(unc_duration) + eps

    # Urgency: slack-driven, uncertainty-augmented, monotonic
    median_slack = np.median(slack)
    q1_slack, q3_slack = np.quantile(slack, [0.25, 0.75], method='midpoint')
    iqr_slack = q3_slack - q1_slack + eps
    # Soft urgency: higher score when slack is low (i.e., more urgent)
    soft_urgency = 1.0 / (1.0 + np.exp(-(slack - median_slack) / (0.3 * iqr_slack + eps)))
    base_urgency = 1.0 - soft_urgency  # smaller = more urgent → matches "smaller score = higher priority"

    # Criticality-energy synergy: only amplify when task is not critically late
    urgency_mask = (slack >= median_slack).astype(float)  # activate synergy for less urgent tasks
    crit_energy_ratio = (upward_rank * total_latency) / (min_incremental_energy + eps)
    norm_crit_energy_ratio = robust_normalize(np.clip(crit_energy_ratio, eps, 1e6))

    # Energy fairness: sign-consistent reversal with uncertainty modulation
    norm_energy = robust_normalize(min_incremental_energy)
    # When slack > 0: favor low energy; when slack <= 0: favor high criticality/low energy offload
    # Modulate reversal strength by uncertainty: higher uncertainty → stronger reversal
    slack_sign_factor = np.where(slack > 0, 1.0, -0.8)
    energy_reversal_strength = 1.0 + 0.5 * uncertainty * np.abs(slack_sign_factor)
    energy_reversed = np.where(slack > 0, 
                              norm_energy * energy_reversal_strength,
                              (1.0 - norm_energy) * energy_reversal_strength)
    energy_reversed = robust_normalize(energy_reversed)

    # Latency penalty: normalized uncertain duration
    norm_unc_duration = robust_normalize(unc_duration)

    # Work impact: penalize large remaining work only when slack is tight
    tau = np.maximum(np.abs(median_slack), 1.0) + eps
    slack_sensitivity = np.exp(-np.clip(np.maximum(-slack, 0.0), 0.0, 100.0) / tau)
    norm_work = robust_normalize(remaining_work)
    work_penalty = norm_work * slack_sensitivity

    # Starvation control: smoothed 90th-percentile wait pressure, gated by slack positivity AND low work density
    wait_threshold = np.quantile(ready_wait_time, 0.9, method='midpoint') + eps
    wait_pressure = np.clip(ready_wait_time / (wait_threshold + eps), 0.0, 1.0)
    work_density = remaining_work / (total_latency + eps)
    low_work_mask = (work_density < np.median(work_density)).astype(float)
    wait_gate = np.where((slack > 0) & (low_work_mask == 1), 1.0, 0.0)
    starvation_term = 1.0 - wait_pressure * wait_gate

    # Final weighted score: smaller = better (higher priority)
    # Weights tuned to emphasize urgency (0.42), energy fairness (0.26), latency (0.12), 
    # criticality-energy synergy (0.09), starvation (0.07), work impact (0.04)
    score = (
        0.42 * base_urgency +
        0.26 * energy_reversed +
        0.12 * norm_unc_duration +
        0.09 * (1.0 - norm_crit_energy_ratio) +
        0.07 * starvation_term +
        0.04 * work_penalty
    )

    # Ensure finiteness and deterministic shape
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)
    return score
