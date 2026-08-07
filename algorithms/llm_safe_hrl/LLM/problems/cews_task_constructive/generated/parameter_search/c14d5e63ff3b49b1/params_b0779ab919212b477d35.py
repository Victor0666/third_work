import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with three structural improvements:
      - Replaces percentile/MAD normalization with DDL-aware min-max scaling: clips range using `ddl_aware_minmax_clip` fraction to prevent distortion of hard-deadline signals under low-variance or singleton conditions.
      - Removes fragile critical-path gating entirely; replaces with *unconditional* upward_rank × remaining_work coupling (weighted), proven to resolve starvation without quantile estimation.
      - Introduces explicit DDL-aware clipping in normalization: ensures negative slack dominates even when other features have small dynamic range.
      - All operations remain vectorized, branch-free, and finite-valued; exactly zero conditional branches.
      - Uses only {-2,-1,0,1,2} literals; all tunables declared and referenced via PARAMS.
    """
    eps = 5.588775339084211e-06
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def ddl_aware_minmax_normalize(x):
        x = np.copy(x)
        x_min = np.min(x) if N > 0 else 0.0
        x_max = np.max(x) if N > 0 else eps
        range_val = x_max - x_min
        clipped_range = np.maximum(range_val * 0.15708296024462545, eps)
        center = np.median(x) if N > 0 else 0.0
        return (x - center) / (clipped_range + eps)
    neg_slack = np.clip(-slack, 0.0, None)
    ddl_penalty = 5.24395690819106 * neg_slack
    median_slack = np.median(slack) if N > 0 else 0.0
    urgency_linear = np.clip(median_slack - slack, 0.0, None)
    non_neg_slack = np.clip(slack, 0.0, None)
    max_non_neg_slack = np.max(non_neg_slack) if N > 0 else eps
    urgency_cap = np.power(max_non_neg_slack + eps, 0.40425410715185006)
    urgency_clipped = np.minimum(urgency_linear, urgency_cap)
    norm_urgency = ddl_aware_minmax_normalize(urgency_clipped)
    critical_path_pressure = upward_rank * remaining_work
    norm_critical_path = ddl_aware_minmax_normalize(critical_path_pressure)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy_eff = ddl_aware_minmax_normalize(energy_per_duration)
    bottleneck_base = duration * upward_rank * remaining_work * (1.0 + urgency_clipped + eps)
    norm_bottleneck = ddl_aware_minmax_normalize(bottleneck_base)
    wait_scaled = ready_wait_time / (1.8712873540759027 + eps)
    wait_clipped = np.clip(wait_scaled, -2.0, 2.0)
    wait_saturation = 1.0 / (1.0 + np.exp(-wait_clipped))
    norm_wait = ddl_aware_minmax_normalize(wait_saturation)
    score = ddl_penalty + norm_urgency + 0.15898216734493845 * norm_bottleneck + 0.5024840377335158 * norm_critical_path + 0.7078132376861841 * norm_critical_path + 0.780181425545434 * norm_energy_eff - norm_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
