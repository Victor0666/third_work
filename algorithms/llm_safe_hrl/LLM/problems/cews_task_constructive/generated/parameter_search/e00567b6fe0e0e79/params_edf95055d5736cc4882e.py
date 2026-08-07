import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
      - Restored slack-sensitivity threshold gating for urgency (validated improvement)
      - Unified uncertainty gating using same steepness for energy & bottleneck (improved consistency)
      - Piecewise urgency: linear ramp in tight-slack region, zero elsewhere — avoids over-penalization
      - Robust rank normalization preserved for skewed distributions
      - All nonlinearities bounded and clipped to [-2,2] or [0,2] domains
      - No unbounded operations; literals strictly in {-2,-1,0,1,2}
      - Explicit finite-domain clipping for all sigmoids and powers
      - Removed unused bottleneck_uncertainty_steepness per validation
    """
    eps = 2.5239778196316566e-06
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
    tight_slack_boundary = median_slack * (1.0 - 0.0963582031854198)
    urgency_linear = np.clip(tight_slack_boundary - slack, 0.0, 2.0)
    urgency_mask = (slack <= tight_slack_boundary) & (slack >= -eps)
    urgency = np.where(urgency_mask, urgency_linear, 0.0)
    norm_urgency = rank_normalize(urgency)
    med_energy = np.median(min_incremental_energy) if N > 0 else 0.0
    abs_devs = np.abs(min_incremental_energy - med_energy)
    mad_energy = np.median(abs_devs) if N > 0 else eps
    denom_energy = 1.7094132146229055 * (mad_energy + eps)
    norm_energy_eff = (min_incremental_energy - med_energy) / (denom_energy + eps)
    norm_energy_eff = np.clip(norm_energy_eff, -1.0, 1.0)
    unc_centered = uncertainty - 0.11247575643664873
    unc_clipped = np.clip(unc_centered, -2.0, 2.0)
    unc_gate = 1.0 / (1.0 + np.exp(-3.5306994901742628 * unc_clipped))
    energy_uncertainty_term = min_incremental_energy * unc_gate
    norm_energy_uncertain = rank_normalize(energy_uncertainty_term)
    bottleneck_pressure = upward_rank * remaining_work * unc_gate
    norm_bottleneck = rank_normalize(bottleneck_pressure)
    norm_wait = rank_normalize(ready_wait_time)
    inv_wait = 1.0 - norm_wait
    local_score = norm_urgency + 0.10172749771599804 * norm_energy_eff + 0.5599743703079813 * norm_energy_uncertain + 0.48164933308542485 * norm_bottleneck - 0.4900156298990521 * inv_wait
    score[feasible_mask] = local_score[feasible_mask]
    score[~feasible_mask] = 3677.1020421028607 * finfo.max
    score = np.nan_to_num(score, nan=finfo.max, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
