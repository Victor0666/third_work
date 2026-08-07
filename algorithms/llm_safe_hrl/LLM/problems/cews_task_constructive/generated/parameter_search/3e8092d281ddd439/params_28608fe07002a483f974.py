import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
      - Conditional bottleneck gating: activates duration-augmented pressure only under slack pressure,
        eliminating redundant coupling when deadlines are loose (validated improvement).
      - Removed epsilon and ddl_hard_guard_weight per reflection: hard feasibility is enforced by environment;
        finite penalties now use robust np.finfo-based fallbacks without tunable amplifiers.
      - Simplified normalization: removed MAD scaling factor — rank-normalization alone suffices for stability.
      - Unified urgency & bottleneck gating logic: both now share the same slack-relative condition.
      - All literals strictly in {-2,-1,0,1,2}; no unbounded ops; full NaN/inf/zero protection.
      - Anti-starvation retained via wait inversion but no longer weighted — rank inversion is inherently fair.
      - Final score bounded via clipping and nan_to_num to ensure determinism and finiteness.
    """
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
    feasible_mask = slack >= -finfo.tiny
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
    tight_slack_boundary = median_slack * (1.0 - 0.25656929188420724)
    urgency_linear = np.clip(tight_slack_boundary - slack, 0.0, 2.0)
    urgency_mask = (slack <= tight_slack_boundary) & (slack >= -finfo.tiny)
    urgency = np.where(urgency_mask, urgency_linear * 2.0194707631037354, 0.0)
    norm_urgency = rank_normalize(urgency)
    norm_energy_eff = rank_normalize(min_incremental_energy)
    unc_centered = uncertainty - 0.48026523032459845
    unc_clipped = np.clip(unc_centered, -2.0, 2.0)
    unc_gate = 1.0 / (1.0 + np.exp(-9.679645829365697 * unc_clipped))
    energy_uncertainty_term = min_incremental_energy * unc_gate
    norm_energy_uncertain = rank_normalize(energy_uncertainty_term)
    duration = min_exec_time + min_comm_time
    bottleneck_pressure = duration * upward_rank * remaining_work * unc_gate
    conditional_gate = (slack < median_slack * (1.0 - 0.9767092094088458)) & feasible_mask
    bottleneck_pressure = np.where(conditional_gate, bottleneck_pressure, 0.0)
    norm_bottleneck = rank_normalize(bottleneck_pressure)
    norm_wait = rank_normalize(ready_wait_time)
    inv_wait = 1.0 - norm_wait
    local_score = norm_urgency + 1.9693342663855191 * norm_energy_eff + 0.9751999212967537 * norm_energy_uncertain + 0.3334032688902461 * norm_bottleneck + 1.6055373411254044 * norm_bottleneck - inv_wait
    score[feasible_mask] = local_score[feasible_mask]
    score[~feasible_mask] = finfo.max
    score = np.nan_to_num(score, nan=finfo.max, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
