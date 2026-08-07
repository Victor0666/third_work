import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining strengths from both parents:
      - Bounded linear urgency ramp on slack (Parent 2) with refined offset for better zero-slack neutrality.
      - Host-load-aware DDL-protection gate using uncertainty × slack-pressure proxy (Parent 2), enhanced with cap on release pressure.
      - Successor-release pressure capped to prevent numerical explosion near zero duration (novel improvement from Parent 1's cap insight).
      - Tighter IQR normalization (22/78 percentiles) for outlier robustness while preserving signal fidelity.
      - Critical-path bonus gated by percentile rank (Parent 2) but computed with safe rank indexing for N=1.
      - Ready wait time uses bounded arctan saturation (Parent 2) with tuned time constant.
      - All operations guarded against NaN/inf/zero; deterministic, finite output.
    """
    eps = 0.0006410668858240382
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def iqr_normalize(x):
        x = np.copy(x)
        q_low = np.percentile(x, 18.124861344786325)
        q_high = np.percentile(x, 83.01687130526658)
        iqr = q_high - q_low
        center = np.median(x)
        denom = iqr if iqr > eps else np.max(np.abs(x - center)) + eps
        return (x - center) / (denom + eps)
    urgency_linear = 5.395160673514372 * (slack - 0.04662472536019102)
    urgency_clamped = np.clip(urgency_linear, -2.0, 2.0)
    norm_urgency = iqr_normalize(urgency_clamped)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / duration
    norm_energy_eff = iqr_normalize(energy_per_duration)
    release_pressure_uncapped = remaining_work * upward_rank / (duration + eps)
    release_pressure = np.minimum(release_pressure_uncapped, 8.823090373368872)
    norm_release = iqr_normalize(release_pressure)
    if N == 1:
        rank_percentile = np.array([1.0])
    else:
        sorted_ranks = np.sort(upward_rank)
        rank_idx = np.searchsorted(sorted_ranks, upward_rank, side='right')
        rank_percentile = rank_idx / (N + eps)
    critical_gate = np.where(rank_percentile >= 0.6875988948900157, 1.0, 0.0)
    norm_rank = iqr_normalize(upward_rank)
    critical_bonus = critical_gate * norm_rank
    wait_scaled = ready_wait_time / (1.355265106459679 + eps)
    wait_saturation = 2.0 / np.pi * np.arctan(wait_scaled)
    norm_wait = iqr_normalize(wait_saturation)
    median_slack = np.median(slack)
    max_uncertainty = np.max(uncertainty) if N > 0 else 1.0
    slack_pressure = np.maximum(0.0, median_slack - slack)
    host_load_proxy = uncertainty * slack_pressure
    ddl_protection_active = ((uncertainty > 0.6126688496260901 * max_uncertainty) & (host_load_proxy > 0.40434035580312533 * np.max(host_load_proxy + eps))).astype(float)
    norm_uncertainty = iqr_normalize(uncertainty)
    score = norm_urgency + 2.2065045255465177 * norm_release + 0.5576643212600534 * norm_energy_eff - critical_bonus - norm_wait + ddl_protection_active * norm_uncertainty
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
