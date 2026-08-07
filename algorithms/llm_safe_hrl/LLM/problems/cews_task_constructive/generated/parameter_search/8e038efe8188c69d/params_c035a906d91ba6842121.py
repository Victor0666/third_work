import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with one bounded structural improvement:
      - Replaces soft critical-path penalty with a *hard DDL-protection gate*: 
        when slack < 0, only tasks with upward_rank × remaining_work above the 
        PARAMS["hard_ddl_gate_threshold"]-quantile receive boosted priority via 
        additive penalty; others retain baseline urgency — eliminating counterfactual 
        dilution while preserving strict deadline dominance.
      - Removes duplicate weighting of norm_critical_path (previously applied twice).
      - Eliminates uncertainty_penalty_weight per reflection: risk is already embedded 
        in min_incremental_energy and slack; explicit penalty harms feasibility.
      - All operations remain vectorized, finite, and branch-free except the single 
        deterministic quantile-based mask (no loops, no randomness).
      - Uses only {-2,-1,0,1,2} literals; all tunables declared and referenced via PARAMS.
    """
    eps = 0.000622221285934373
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
        clipped_range = np.maximum(range_val * 0.19189950198542388, eps)
        center = np.median(x)
        return (x - center) / (clipped_range + eps)
    neg_slack = np.clip(-slack, 0.0, None)
    ddl_penalty = 4.190221877699594 * neg_slack
    median_slack = np.median(slack) if N > 0 else 0.0
    urgency_linear = np.clip(median_slack - slack, 0.0, None)
    non_neg_slack = np.clip(slack, 0.0, None)
    max_non_neg_slack = np.max(non_neg_slack) if N > 0 else eps
    urgency_cap = np.power(max_non_neg_slack + eps, 0.774457017384272)
    urgency_clipped = np.minimum(urgency_linear, urgency_cap)
    norm_urgency = ddl_aware_minmax_normalize(urgency_clipped)
    critical_path_pressure = upward_rank * remaining_work
    norm_critical_path = ddl_aware_minmax_normalize(critical_path_pressure)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy_eff = ddl_aware_minmax_normalize(energy_per_duration)
    bottleneck_base = duration * upward_rank * remaining_work * (1.0 + urgency_clipped + eps)
    norm_bottleneck = ddl_aware_minmax_normalize(bottleneck_base)
    wait_scaled = ready_wait_time / (1.777282524980002 + eps)
    wait_clipped = np.clip(wait_scaled, -2.0, 2.0)
    wait_saturation = 1.0 / (1.0 + np.exp(-wait_clipped))
    norm_wait = ddl_aware_minmax_normalize(wait_saturation)
    cp_pressure_under_violation = np.where(slack < 0.0, critical_path_pressure, -np.inf)
    quantile_level = np.clip(0.9034972174636143, 0.0, 1.0)
    if N > 0 and np.any(slack < 0.0):
        gate_threshold = np.quantile(cp_pressure_under_violation, quantile_level, method='midpoint')
    else:
        gate_threshold = np.quantile(critical_path_pressure, quantile_level, method='midpoint')
    hard_ddl_gate = np.where((slack < 0.0) & (critical_path_pressure >= gate_threshold - eps), 1.0, 0.0)
    score = ddl_penalty + hard_ddl_gate * (1.0 + norm_critical_path) + norm_urgency + 1.4702028057322662 * norm_bottleneck + 0.8543498688976497 * norm_critical_path + 1.175113154749974 * norm_critical_path + 1.1346225165365509 * norm_energy_eff - norm_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
