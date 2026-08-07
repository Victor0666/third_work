import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with adaptive deadline-criticality detection and MAD-based robust scaling.
    
    Key structural improvements:
      - Replaces fixed `ddl_feasibility_threshold` with percentile-based adaptive criticality detection:
        tasks in bottom `slack_critical_percentile` of slack distribution enter DDL-stress mode.
      - Introduces multiplicative `upward_rank * remaining_work` coupling *only* under DDL stress,
        replacing linear bottleneck sum — better captures joint critical-path + work-pressure amplification.
      - Switches `slack` and `uncertainty` normalization from IQR to MAD scaling (with consistent 1.4826 factor)
        for superior outlier resilience and stability across heterogeneous workflows.
      - Removes redundant `ddl_protection_mask` duplication; unifies risk logic into single adaptive urgency path.
      - All non-urgency terms are suppressed *only* under adaptive DDL stress (not hard threshold), improving robustness.
      - Retains arctan wait saturation, host-load–aware energy, and safe percentile ranking for N=1.
    """
    eps = 1e-06
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def mad_normalize(x):
        x = np.copy(x)
        center = np.median(x)
        abs_devs = np.abs(x - center)
        mad = np.median(abs_devs)
        scale = 1.73737713497902 * mad
        denom = scale if scale > eps else np.max(abs_devs) + eps
        return (x - center) / (denom + eps)
    if N == 1:
        slack_percentile = np.array([0.0])
    else:
        sorted_slack = np.sort(slack)
        slack_idx = np.searchsorted(sorted_slack, slack, side='left')
        slack_percentile = slack_idx / (N + eps)
    ddl_stress_mask = (slack_percentile < 0.2727651579509354).astype(float)
    median_abs_slack = np.maximum(np.abs(np.median(slack)), eps)
    urgency_gate = np.clip(slack / (median_abs_slack + eps), -1.0, 1.0)
    urgency_gate = -urgency_gate
    urgency_amplified = urgency_gate * (1.0 + 3.029153937959297 * ddl_stress_mask)
    norm_urgency = mad_normalize(urgency_amplified)
    host_load_adjusted_energy = min_incremental_energy * (1.0 + 0.34137950526048016 * uncertainty)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = host_load_adjusted_energy / duration
    norm_energy_eff = mad_normalize(energy_per_duration) * (1.0 - ddl_stress_mask)
    rank_work_pressure = (upward_rank + eps) * (remaining_work + eps)
    norm_rank_work = mad_normalize(rank_work_pressure) * ddl_stress_mask
    if N == 1:
        rank_percentile = np.array([1.0])
    else:
        sorted_ranks = np.sort(upward_rank)
        rank_idx = np.searchsorted(sorted_ranks, upward_rank, side='right')
        rank_percentile = rank_idx / (N + eps)
    critical_gate = np.where(rank_percentile >= 0.8589620597325869, 1.0, 0.0)
    norm_rank = mad_normalize(upward_rank)
    critical_bonus = critical_gate * norm_rank * (1.0 - ddl_stress_mask)
    wait_scaled = ready_wait_time / (9.255270696942173 + eps)
    wait_saturation = 2.0 / np.pi * np.arctan(wait_scaled)
    norm_wait = mad_normalize(wait_saturation) * (1.0 - ddl_stress_mask)
    norm_uncertainty = mad_normalize(uncertainty)
    risk_amplification_mask = ((slack < 0) & (uncertainty > 0.8427425758940098 * np.max(uncertainty + eps))).astype(float)
    risk_amplified_urgency = norm_urgency * (1.0 + 0.4333361404586685 * norm_uncertainty * risk_amplification_mask)
    score = risk_amplified_urgency + 0.15178745912334596 * norm_rank_work + 0.4375637167097437 * norm_energy_eff - critical_bonus - norm_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
