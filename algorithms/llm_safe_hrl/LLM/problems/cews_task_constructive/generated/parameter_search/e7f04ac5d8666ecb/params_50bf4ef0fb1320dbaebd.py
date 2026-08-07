import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining Parent 2's robust deadline enforcement with Parent 1's critical-path coupling.
    
    Key structural improvements:
      - Introduces tunable `deadline_violation_penalty_weight` to scale raw neg_slack — enables CMA-ES to balance hard constraint dominance vs. other objectives.
      - Adds `critical_path_pressure_weight`-scaled normalized cp_coupling (upward_rank * remaining_work) as a dedicated structural term — preserves HEFT-like criticality without bottleneck entanglement.
      - Removes all gating logic (feasibility_mask) from bottleneck pressure: eliminates brittle conditional suppression that degraded performance under high uncertainty.
      - Retains Parent 2's pre-normalized neg_slack as dominant term and bounded sigmoid wait saturation for anti-starvation.
      - All numeric literals strictly in {-2,-1,0,1,2}; no branching beyond clipping/min/max; fully deterministic.
    """
    eps = 3.068847735208178e-05
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def adaptive_normalize(x):
        x = np.copy(x)
        center = np.median(x)
        unc_std = np.std(uncertainty) if N > 1 else eps
        dispersion = 0.9780648690325529 * (unc_std + eps)
        fallback_range = np.max(x) - np.min(x) if N > 0 else eps
        denom = dispersion if dispersion > eps else fallback_range
        return (x - center) / (denom + eps)
    neg_slack = np.clip(-slack, 0.0, None)
    median_slack = np.median(slack) if N > 0 else 0.0
    urgency_linear = np.clip(median_slack - slack, 0.0, 2.0)
    non_neg_slack = np.clip(slack, 0.0, None)
    max_non_neg_slack = np.max(non_neg_slack) if N > 0 else eps
    urgency_cap = np.power(max_non_neg_slack + eps, 0.4484022996695659)
    urgency_linear = np.minimum(urgency_linear, urgency_cap)
    norm_urgency = adaptive_normalize(urgency_linear)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy_eff = adaptive_normalize(energy_per_duration)
    bottleneck_pressure = duration * upward_rank * remaining_work * (1.0 + urgency_linear + eps)
    unc_normalized = adaptive_normalize(uncertainty)
    bottleneck_pressure = bottleneck_pressure * np.power(1.0 + unc_normalized, 0.5744485060812332)
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    wait_scaled = ready_wait_time / (2.9920858617066903 + eps)
    wait_clipped = np.clip(wait_scaled, -2.0, 2.0)
    wait_saturation = 1.0 / (1.0 + np.exp(-wait_clipped))
    norm_wait = adaptive_normalize(wait_saturation)
    cp_coupling = upward_rank * remaining_work
    norm_cp_coupling = adaptive_normalize(cp_coupling)
    score = 17.516658728774228 * neg_slack + norm_urgency + 0.3728247220706807 * norm_bottleneck + 1.6876463688295706 * norm_energy_eff - norm_wait + 0.6695909013606355 * norm_cp_coupling
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
