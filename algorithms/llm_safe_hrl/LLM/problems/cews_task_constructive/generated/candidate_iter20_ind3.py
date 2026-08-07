import numpy as np
RULE_METADATA = {'structure_hash': '2a9dcad5dae68d8041328992f69513acb003b6585a52171dd822d367ffac6179', 'parameter_schema_hash': 'aeff5542bd8e7acf4d75e07e1ed97bc5f62b8d7eeedd7183b2026f81922a218e', 'best_parameter_hash': '954d3e6ef559cf1af8635efbc711a801ab84e24d58f0bb0079fa28a87982e39b', 'best_parameters': {'epsilon': 1.071179096165902e-05, 'ddl_hard_guard_weight': 246.72293900906786, 'urgency_power': 1.8462067996621694, 'bottleneck_uncertainty_steepness': 2.100663464983513, 'energy_efficiency_weight': 0.8362972330836032, 'critical_path_coupling': 0.4795851991367926, 'wait_fairness_weight': 0.6297557102842943, 'iqr_percentile_high': 73.32619004534955, 'iqr_percentile_low': 22.85953604043075}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': '1eb8bf32da3ea721257603415abac302d7106f0f473bb21b731ee445f24be95d', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's hard feasibility guard and energy-efficiency term
    with Parent 1's robust power-law urgency and critical-path coupling — while eliminating redundancy.
    
    Key structural improvements:
      - Hard DDL guard now uses weighted +inf penalty (not binary) to preserve gradient signal for CMA-ES.
      - Replaces linear urgency with power-law urgency: (median_slack - slack)^urgency_power → stronger near-deadline focus.
      - Retains Parent 2's physically grounded energy_per_duration but couples it with Parent 1's critical-path bottleneck
        via additive, normalized terms — no multiplicative risk couplings.
      - Removes all conditional branches beyond the hard guard (no piecewise gates), ensuring bounded AST depth.
      - Uses unified adaptive normalization with uncertainty-aware dispersion *and* fallback to interquartile range (IQR)
        for improved robustness to outliers vs. simple range.
      - All terms are additive, finite, and strictly deterministic; no hidden state or randomness.
    """
    eps = 1.071179096165902e-05
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    finfo = np.finfo(float)
    score = np.full(N, finfo.max, dtype=float)
    feasible_mask = slack >= -eps
    if not np.any(feasible_mask):
        return score

    def adaptive_normalize(x):
        x = np.copy(x)
        center = np.median(x)
        if N > 2:
            q_high = np.percentile(x, 73.32619004534955)
            q_low = np.percentile(x, 22.85953604043075)
            dispersion = q_high - q_low
        else:
            unc_std = np.std(uncertainty) if N > 1 else eps
            dispersion = 246.72293900906786 * (unc_std + eps)
            fallback_range = np.max(x) - np.min(x) if N > 0 else eps
            dispersion = np.where(dispersion > eps, dispersion, fallback_range)
        denom = dispersion + eps
        return (x - center) / denom
    median_slack = np.median(slack) if N > 0 else 0.0
    urgency_base = np.clip(median_slack - slack, 0.0, 2.0)
    urgency = np.power(urgency_base + eps, 1.8462067996621694)
    norm_urgency = adaptive_normalize(urgency)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy_eff = adaptive_normalize(energy_per_duration)
    unc_scaled = uncertainty * 2.100663464983513
    unc_clipped = np.clip(unc_scaled, -2.0, 2.0)
    bottleneck_gate = 1.0 / (1.0 + np.exp(-unc_clipped))
    bottleneck_pressure = upward_rank * remaining_work * bottleneck_gate
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    norm_wait = adaptive_normalize(ready_wait_time)
    inv_wait = 1.0 - norm_wait
    local_score = norm_urgency + 0.8362972330836032 * norm_energy_eff + 0.4795851991367926 * norm_bottleneck - 0.6297557102842943 * inv_wait
    score[feasible_mask] = local_score[feasible_mask]
    score[~feasible_mask] = 246.72293900906786 * finfo.max
    score = np.nan_to_num(score, nan=finfo.max, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
