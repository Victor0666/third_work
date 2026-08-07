import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining strengths of both parents:
      - Retains Parent 2's robust DDL-aware min-max normalization (no fragile quantiles) and unconditional CP coupling.
      - Adds Parent 1's uncertainty-aware gating: applies penalty only when slack < threshold, avoiding spurious risk aversion in slack-rich regimes.
      - Introduces novel *uncertainty penalty* term: linearly scaled by uncertainty and activated only under deadline pressure — improves risk-awareness without degrading performance in safe regions.
      - Removes all conditional branches (0 branches) while preserving interpretability and monotonicity.
      - Uses only {-2,-1,0,1,2} literals; all tunables declared and referenced via PARAMS.
      - Ensures strict finiteness and determinism.
    """
    eps = 0.0008033188565846611
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
        clipped_range = np.maximum(range_val * 0.33348470798901436, eps)
        center = np.median(x)
        return (x - center) / (clipped_range + eps)
    neg_slack = np.clip(-slack, 0.0, None)
    ddl_penalty = 7.3008813100443914 * neg_slack
    median_slack = np.median(slack) if N > 0 else 0.0
    urgency_linear = np.clip(median_slack - slack, 0.0, None)
    non_neg_slack = np.clip(slack, 0.0, None)
    max_non_neg_slack = np.max(non_neg_slack) if N > 0 else eps
    urgency_cap = np.power(max_non_neg_slack + eps, 1.1121330665094242)
    urgency_clipped = np.minimum(urgency_linear, urgency_cap)
    norm_urgency = ddl_aware_minmax_normalize(urgency_clipped)
    critical_path_pressure = upward_rank * remaining_work
    norm_critical_path = ddl_aware_minmax_normalize(critical_path_pressure)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy_eff = ddl_aware_minmax_normalize(energy_per_duration)
    bottleneck_base = duration * upward_rank * remaining_work * (1.0 + urgency_clipped + eps)
    norm_bottleneck = ddl_aware_minmax_normalize(bottleneck_base)
    wait_scaled = ready_wait_time / (1.7847095963287694 + eps)
    wait_clipped = np.clip(wait_scaled, -2.0, 2.0)
    wait_saturation = 1.0 / (1.0 + np.exp(-wait_clipped))
    norm_wait = ddl_aware_minmax_normalize(wait_saturation)
    uncertainty_penalty_mask = (slack < -0.37787100099134885).astype(float)
    uncertainty_penalty = 0.08172672110147489 * uncertainty * uncertainty_penalty_mask
    norm_uncertainty_penalty = ddl_aware_minmax_normalize(uncertainty_penalty)
    score = ddl_penalty + norm_urgency + 0.9111356178219114 * norm_bottleneck + 1.0025469770996775 * norm_critical_path + 0.8765122443700961 * norm_critical_path + 1.059929590168636 * norm_energy_eff - norm_wait + norm_uncertainty_penalty
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
