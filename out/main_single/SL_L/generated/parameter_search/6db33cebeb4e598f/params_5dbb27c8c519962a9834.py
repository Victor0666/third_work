import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with two key structural improvements:
      - Replaces brittle `slack > 0.0` lexicographic gate with smooth, differentiable sigmoid feasibility gate:
        `1 / (1 + exp(-steepness * slack))`, enabling gradient-based optimization and robust interpolation.
      - Switches from percentile to rank-based (ordinal) normalization: computes rank order then maps to [0,1] via `(rank - 1) / (N - 1)` for N>1,
        eliminating degeneracy in flat/duplicate distributions while preserving monotonicity and boundedness.
      - Load-aware wait mitigation is retained but simplified: uses fixed median threshold (no new parameter) and couples directly with feasibility_gate.
    """
    eps = 1.0860025339047538e-07
    finfo = np.finfo(float)
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=finfo.max, neginf=finfo.min)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=finfo.max, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=eps, posinf=finfo.max, neginf=eps)
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)

    def rank_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        sorted_idx = np.argsort(-x)
        ranks = np.empty_like(sorted_idx, dtype=float)
        ranks[sorted_idx] = np.arange(1, N + 1)
        return (ranks - 1.0) / (N - 1.0)
    slack_score = np.where(slack < 0, (-slack) ** 1.9376020397202005, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 1.6407551399532465
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 1.4547119098858854
    is_tight_or_violated = slack <= 0
    rank_median = np.median(upward_rank) if N > 1 else np.mean(upward_rank)
    work_median = np.median(remaining_work) if N > 1 else np.mean(remaining_work)
    is_high_rank = upward_rank >= rank_median
    is_high_work = remaining_work >= work_median
    critical_gate = np.where(is_high_rank & is_high_work & is_tight_or_violated, 0.6437915519593531, 1.0)
    feasibility_gate = 1.0 / (1.0 + np.exp(-0.5700331059670044 * slack))
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = rank_normalize(energy_per_sec)
    slack_lb = -16.654533604141676
    slack_ub = 23.29716951732917
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.5724171955287602 + (1.0 - 0.5724171955287602) * (1.0 - slack_scaled)
    rank_score = -rank_normalize(upward_rank) * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-1.7462349598519036 * (uncertainty - 1.0)))
    energy_norm = rank_normalize(min_incremental_energy)
    unc_norm = rank_normalize(uncertainty)
    energy_uncertainty_score = 1.0672220538170245 * energy_norm * unc_norm * unc_sigmoid
    wait_median = np.median(ready_wait_time) if N > 1 else np.mean(ready_wait_time)
    load_aware_mask = np.where(ready_wait_time > wait_median, 1.0, 0.0)
    load_aware_score = load_aware_mask * energy_eff_score * feasibility_gate
    score = rank_normalize(slack_score) + rank_normalize(unc_slack_coupling) + rank_normalize(duration_risk)
    score += feasibility_gate * (0.7682664885634529 * energy_eff_score + rank_score + energy_uncertainty_score + load_aware_score)
    score = score * critical_gate
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
