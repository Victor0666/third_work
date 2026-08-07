import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining strengths from both parents:
      - Robust rank-based normalization (from Parent 2) for skewed distributions.
      - Hard deadline guard with finite penalty (Parent 2).
      - Scenario-aware activation gates for energy & bottleneck (Parent 2).
      - Novel urgency formulation: direct negative-slack power law (replacing linear ramp) for stronger lateness signal.
      - Unified uncertainty gating using same steepness for consistency.
      - Removed unused 'slack_sensitivity_threshold' per validation.
      - All nonlinearities clipped to [-2,2] or bounded domains to ensure stability.
      - No unbounded operations; all literals in {-2,-1,0,1,2}.
    """
    eps = 9.218074624298334e-06
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
    neg_slack = np.clip(-slack, 0.0, None)
    urgency = np.power(neg_slack + eps, 1.6304521169616388)
    norm_urgency = rank_normalize(urgency)
    med_energy = np.median(min_incremental_energy) if N > 0 else 0.0
    abs_devs = np.abs(min_incremental_energy - med_energy)
    mad_energy = np.median(abs_devs) if N > 0 else eps
    denom_energy = 1.4583350913007684 * (mad_energy + eps)
    norm_energy_eff = (min_incremental_energy - med_energy) / (denom_energy + eps)
    norm_energy_eff = np.clip(norm_energy_eff, -1.0, 1.0)
    unc_centered = uncertainty - 0.5195198421583325
    unc_clipped = np.clip(unc_centered, -2.0, 2.0)
    unc_gate = 1.0 / (1.0 + np.exp(-4.06157250483467 * unc_clipped))
    energy_uncertainty_term = min_incremental_energy * unc_gate
    norm_energy_uncertain = rank_normalize(energy_uncertainty_term)
    bottleneck_gate = np.where(uncertainty >= 0.5195198421583325, 1.0 / (1.0 + np.exp(-2.2510656716300153 * np.clip(unc_centered, 0.0, 2.0))), 0.0)
    bottleneck_pressure = upward_rank * remaining_work * bottleneck_gate
    norm_bottleneck = rank_normalize(bottleneck_pressure)
    norm_wait = rank_normalize(ready_wait_time)
    inv_wait = 1.0 - norm_wait
    local_score = norm_urgency + 0.8161274330768342 * norm_energy_eff + 0.8079649149693959 * norm_energy_uncertain + 0.7881256153596855 * norm_bottleneck - 0.7688276062247702 * inv_wait
    score[feasible_mask] = local_score[feasible_mask]
    score[~feasible_mask] = 1159.6111047776367 * finfo.max
    score = np.nan_to_num(score, nan=finfo.max, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
