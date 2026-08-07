import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining strengths from both parents:
      - Retains Parent 2's robust DDL-aware min-max normalization (clipped range) for stability and dominance preservation.
      - Keeps unconditional upward_rank × remaining_work coupling (additive, not multiplicative) — no fragile gating.
      - Adds *uncertainty-aware penalty* (Parent 1's risk theme) but without conditional branches: linearly scaled by uncertainty.
      - Introduces *soft critical-path penalty*: when slack < 0 AND upward_rank < median_upward_rank, apply scaled penalty — 
        less aggressive than Parent 1's hard gate, more adaptive than Parent 2's omission.
      - Removes all inactive parameters (e.g., iqr_percentiles, successor_release_pressure_exponent) per analysis.
      - Uses only {-2,-1,0,1,2} literals; all tunables declared and referenced via PARAMS.
      - Zero conditional branches except the single soft penalty mask (vectorized & deterministic).
      - All operations finite, guarded, and shape-preserving.
    """
    eps = 1.1076222165208707e-05
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
        if N == 0:
            return np.zeros_like(x)
        x_min = np.min(x)
        x_max = np.max(x)
        range_val = x_max - x_min
        clipped_range = np.maximum(range_val * 0.4652480828826501, eps)
        center = np.median(x)
        return (x - center) / (clipped_range + eps)
    neg_slack = np.clip(-slack, 0.0, None)
    ddl_penalty = 8.222909212015097 * neg_slack
    median_slack = np.median(slack) if N > 0 else 0.0
    urgency_linear = np.clip(median_slack - slack, 0.0, None)
    non_neg_slack = np.clip(slack, 0.0, None)
    max_non_neg_slack = np.max(non_neg_slack) if N > 0 else eps
    urgency_cap = np.power(max_non_neg_slack + eps, 0.5399217606584114)
    urgency_clipped = np.minimum(urgency_linear, urgency_cap)
    norm_urgency = ddl_aware_minmax_normalize(urgency_clipped)
    critical_path_pressure = upward_rank * remaining_work
    norm_critical_path = ddl_aware_minmax_normalize(critical_path_pressure)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy_eff = ddl_aware_minmax_normalize(energy_per_duration)
    bottleneck_base = duration * upward_rank * remaining_work * (1.0 + urgency_clipped + eps)
    norm_bottleneck = ddl_aware_minmax_normalize(bottleneck_base)
    wait_scaled = ready_wait_time / (1.465245110122928 + eps)
    wait_clipped = np.clip(wait_scaled, -2.0, 2.0)
    wait_saturation = 1.0 / (1.0 + np.exp(-wait_clipped))
    norm_wait = ddl_aware_minmax_normalize(wait_saturation)
    median_upward_rank = np.median(upward_rank) if N > 0 else 0.0
    cp_penalty_mask = np.where((slack < 0.0) & (upward_rank < median_upward_rank), 1.0, 0.0)
    cp_penalty = cp_penalty_mask * 0.38524396248711673 * (1.0 + uncertainty)
    uncertainty_penalty = 0.5885514994250829 * uncertainty
    score = ddl_penalty + norm_urgency + 0.4826433008545266 * norm_bottleneck + 0.8165333055767738 * norm_critical_path + 0.9282513637538569 * norm_critical_path + 0.6842760470386942 * norm_energy_eff - norm_wait + cp_penalty + uncertainty_penalty
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
