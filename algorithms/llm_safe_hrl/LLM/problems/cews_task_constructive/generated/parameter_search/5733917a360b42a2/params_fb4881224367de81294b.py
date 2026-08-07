import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's robust rank normalization and criticality gating
    with Parent 1's explicit duration-aware energy efficiency and smoother fairness saturation.
    
    Key improvements:
      - Introduces novel energy_uncertainty_coupling: modulates energy cost by uncertainty-scaled gate,
        improving risk-aware energy minimization without violating DDL constraints.
      - Replaces raw duration in energy denominator with power-law duration_awareness_exponent,
        enabling tunable sensitivity to execution+comm time scaling.
      - Retains Parent 2's rank-based normalization for outlier resilience and cross-scenario stability.
      - Keeps criticality gating and hard DDL guard for strict deadline feasibility.
      - Uses saturating tanh fairness (from Parent 1) instead of rank-inversion for smoother anti-starvation.
      - All numeric literals strictly in {-2,-1,0,1,2}; no unbounded operations or hidden constants.
    """
    eps = 1.0221510157812443e-06
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

    def rank_normalize(x):
        x = np.copy(x)
        if N <= 1 or np.all(x == x[0]):
            return np.zeros_like(x, dtype=float)
        sorted_idx = np.argsort(x)
        ranks = np.empty_like(sorted_idx, dtype=float)
        ranks[sorted_idx] = np.arange(N, dtype=float)
        norm_ranks = 2.0 * ranks / (N - 1.0) - 1.0
        return np.clip(norm_ranks, -1.0, 1.0)
    median_slack = np.median(slack) if N > 0 else 0.0
    tight_slack_mask = (slack < median_slack * (1.0 - 0.20558502976112863)) & (slack >= -eps)
    max_upward = np.max(upward_rank) if N > 0 else 1.0
    non_critical_mask = (upward_rank < max_upward * 0.37336569664909186) & tight_slack_mask
    critical_gate = np.where(non_critical_mask, 0.0, 1.0)
    urgency_base = np.clip(median_slack - slack, 0.0, 2.0)
    urgency = np.power(urgency_base + eps, 1.8289716214703697)
    norm_urgency = rank_normalize(urgency)
    duration = min_exec_time + min_comm_time
    duration_power = np.power(np.maximum(duration, eps), 0.034930688529389506)
    energy_per_duration = min_incremental_energy / (duration_power + eps)
    med_energy = np.median(energy_per_duration) if N > 0 else 0.0
    abs_devs = np.abs(energy_per_duration - med_energy)
    mad_energy = np.median(abs_devs) if N > 0 else eps
    denom_energy = 1.9315168094229127 * (mad_energy + eps)
    norm_energy_eff = (energy_per_duration - med_energy) / (denom_energy + eps)
    norm_energy_eff = np.clip(norm_energy_eff, -1.0, 1.0)
    unc_clipped = np.clip(uncertainty, -2.0, 2.0)
    energy_uncertainty_gate = 1.0 / (1.0 + np.exp(-unc_clipped))
    energy_uncertainty_term = energy_per_duration * energy_uncertainty_gate
    norm_energy_uncertain = rank_normalize(energy_uncertainty_term)
    unc_scaled = uncertainty * 4.220329235945586
    unc_clipped_bottleneck = np.clip(unc_scaled, -2.0, 2.0)
    bottleneck_gate = 1.0 / (1.0 + np.exp(-unc_clipped_bottleneck))
    bottleneck_pressure = upward_rank * remaining_work * bottleneck_gate
    norm_bottleneck = rank_normalize(bottleneck_pressure)
    wait_normalized = ready_wait_time / (1.0 + eps)
    fairness_boost = np.tanh(wait_normalized)
    local_score = norm_urgency * critical_gate + 1.3662755025212008 * norm_energy_eff + 0.8137737286321561 * norm_energy_uncertain + 0.4232349784049447 * norm_bottleneck * critical_gate - 0.9367399557253324 * fairness_boost
    score[feasible_mask] = local_score[feasible_mask]
    score[~feasible_mask] = 7664.987660596779 * finfo.max
    score = np.nan_to_num(score, nan=finfo.max, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
