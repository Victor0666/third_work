import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule featuring:
      - Bounded linear urgency ramp on slack (replaces sigmoid) for improved gradient signal near zero.
      - Host-load-aware DDL-protection gate: activates only when both uncertainty AND host load are high.
      - Direct successor-release pressure: (remaining_work * upward_rank) / (duration + ε), penalizing bottlenecks.
      - Tighter IQR normalization bounds (20/80) for robustness to outliers.
      - All operations guarded against NaN/inf/zero; deterministic and finite output.
    """
    eps = 0.0958509491322721
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
        q_low = np.percentile(x, 9.66839631918523)
        q_high = np.percentile(x, 71.67943993077468)
        iqr = q_high - q_low
        center = np.median(x)
        denom = iqr if iqr > eps else np.max(np.abs(x - center)) + eps
        return (x - center) / (denom + eps)
    urgency_linear = 8.551674866531181 * (slack - -0.6514019153364916)
    urgency_clamped = np.clip(urgency_linear, -2.0, 2.0)
    norm_urgency = iqr_normalize(urgency_clamped)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / duration
    norm_energy_eff = iqr_normalize(energy_per_duration)
    release_pressure = remaining_work * upward_rank / (duration + eps)
    norm_release = iqr_normalize(release_pressure)
    if N == 1:
        rank_percentile = np.array([1.0])
    else:
        sorted_ranks = np.sort(upward_rank)
        rank_idx = np.searchsorted(sorted_ranks, upward_rank, side='right')
        rank_percentile = rank_idx / (N + eps)
    critical_gate = np.where(rank_percentile >= 0.6447327463524826, 1.0, 0.0)
    norm_rank = iqr_normalize(upward_rank)
    critical_bonus = critical_gate * norm_rank
    wait_scaled = ready_wait_time / (21.733684813259913 + eps)
    wait_saturation = 2.0 / np.pi * np.arctan(wait_scaled)
    norm_wait = iqr_normalize(wait_saturation)
    median_slack = np.median(slack)
    max_uncertainty = np.max(uncertainty) if N > 0 else 1.0
    slack_pressure = np.maximum(0.0, median_slack - slack)
    host_load_proxy = uncertainty * slack_pressure
    ddl_protection_active = ((uncertainty > 0.6090488591630789 * max_uncertainty) & (host_load_proxy > 0.7161001125148512 * np.max(host_load_proxy + eps))).astype(float)
    norm_uncertainty = iqr_normalize(uncertainty)
    score = norm_urgency + 1.302648658972102 * norm_release + 0.7248076708002974 * norm_energy_eff - critical_bonus - norm_wait + ddl_protection_active * norm_uncertainty
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
