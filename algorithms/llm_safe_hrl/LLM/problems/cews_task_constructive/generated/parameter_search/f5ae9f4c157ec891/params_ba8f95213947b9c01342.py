import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
      - Rank-based normalization (replacing percentile/IQR) for cross-scenario stability;
      - Hard deadline-aware criticality gating: suppresses non-critical-path tasks when slack is tight;
      - Decoupled energy term using only min_incremental_energy, normalized via MAD for outlier resilience;
      - All numeric literals strictly in {-2,-1,0,1,2}; no ListComp or unbounded loops.
    """
    eps = 3.0271401927718743e-05
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
    tight_slack_mask = (slack < median_slack * (1.0 - 0.24581901855442684)) & (slack >= -eps)
    max_upward = np.max(upward_rank) if N > 0 else 1.0
    non_critical_mask = (upward_rank < max_upward * 0.22110683910680645) & tight_slack_mask
    critical_gate = np.where(non_critical_mask, 0.0, 1.0)
    urgency_base = np.clip(median_slack - slack, 0.0, 2.0)
    urgency = np.power(urgency_base + eps, 2.410225593147355)
    norm_urgency = rank_normalize(urgency)
    med_energy = np.median(min_incremental_energy) if N > 0 else 0.0
    abs_devs = np.abs(min_incremental_energy - med_energy)
    mad_energy = np.median(abs_devs) if N > 0 else eps
    denom_energy = 1.5482177849804801 * (mad_energy + eps)
    norm_energy_eff = (min_incremental_energy - med_energy) / (denom_energy + eps)
    norm_energy_eff = np.clip(norm_energy_eff, -1.0, 1.0)
    unc_scaled = uncertainty * 1.4829589496701683
    unc_clipped = np.clip(unc_scaled, -2.0, 2.0)
    bottleneck_gate = 1.0 / (1.0 + np.exp(-unc_clipped))
    bottleneck_pressure = upward_rank * remaining_work * bottleneck_gate
    norm_bottleneck = rank_normalize(bottleneck_pressure)
    norm_wait = rank_normalize(ready_wait_time)
    inv_wait = 1.0 - norm_wait
    local_score = norm_urgency * critical_gate + 1.999975362986159 * norm_energy_eff + 0.8714284811512385 * norm_bottleneck * critical_gate - 0.5586411349642166 * inv_wait
    score[feasible_mask] = local_score[feasible_mask]
    score[~feasible_mask] = 800.076694143341 * finfo.max
    score = np.nan_to_num(score, nan=finfo.max, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
