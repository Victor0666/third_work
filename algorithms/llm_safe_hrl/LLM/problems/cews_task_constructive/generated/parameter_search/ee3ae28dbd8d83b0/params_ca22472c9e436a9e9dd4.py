import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule featuring:
      - Adaptive DDL-feasibility gating: when slack < ddl_feasibility_threshold, all non-urgency terms are suppressed
        and urgency is amplified via feasibility_guard_strength — enforcing hard deadline compliance first.
      - Simplified urgency: direct slack sign-and-magnitude signal instead of linear ramp; avoids parameter overfitting.
      - Robust IQR normalization with tightened percentiles (21/79) for better outlier resilience.
      - Capped successor-release pressure to prevent numerical instability near zero duration.
      - Critical-path bonus remains percentile-gated but computed safely for N=1.
      - All operations guarded against NaN/inf/zero; deterministic and finite output.
    """
    eps = 0.0005051055391774103
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
        q_low = np.percentile(x, 35.598190341562585)
        q_high = np.percentile(x, 77.66112141413096)
        iqr = q_high - q_low
        center = np.median(x)
        denom = iqr if iqr > eps else np.max(np.abs(x - center)) + eps
        return (x - center) / (denom + eps)
    ddl_critical_mask = (slack < 0.9283579582227395).astype(float)
    urgency_raw = -slack
    urgency_amplified = urgency_raw * (1.0 + 3.921554965464502 * ddl_critical_mask)
    norm_urgency = iqr_normalize(urgency_amplified)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / duration
    norm_energy_eff = iqr_normalize(energy_per_duration)
    release_pressure_uncapped = remaining_work * upward_rank / (duration + eps)
    release_pressure = np.minimum(release_pressure_uncapped, 1.67029761521183)
    norm_release = iqr_normalize(release_pressure)
    if N == 1:
        rank_percentile = np.array([1.0])
    else:
        sorted_ranks = np.sort(upward_rank)
        rank_idx = np.searchsorted(sorted_ranks, upward_rank, side='right')
        rank_percentile = rank_idx / (N + eps)
    critical_gate = np.where(rank_percentile >= 0.8300292641951117, 1.0, 0.0)
    norm_rank = iqr_normalize(upward_rank)
    critical_bonus = critical_gate * norm_rank
    median_slack = np.median(slack)
    max_uncertainty = np.max(uncertainty) if N > 0 else 1.0
    slack_pressure = np.maximum(0.0, median_slack - slack)
    host_load_proxy = uncertainty * slack_pressure
    ddl_protection_active = ((uncertainty > 0.5998324493877597 * max_uncertainty) & (host_load_proxy > 0.6457042264107723 * np.max(host_load_proxy + eps))).astype(float)
    norm_uncertainty = iqr_normalize(uncertainty)
    score = norm_urgency + ddl_critical_mask * 0.0 + ddl_critical_mask * 0.0 + ddl_critical_mask * 0.0 + ddl_critical_mask * 0.0 + (1.0 - ddl_critical_mask) * (0.326801312735689 * norm_energy_eff + 1.161214329492956 * norm_release - critical_bonus + ddl_protection_active * norm_uncertainty)
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
