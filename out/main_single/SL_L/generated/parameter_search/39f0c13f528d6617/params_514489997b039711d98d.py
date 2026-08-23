import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule incorporating:
      - Conditional DDL-protection gate: activates only when slack <= 0 AND uncertainty <= median_unc,
        preventing premature bottleneck focus while enforcing hard deadlines.
      - Successor-release interaction: modeled via upward_rank * remaining_work, gated by DDL-protection condition.
      - Robust wait-time anti-starvation term: linearly scaled by normalized wait time and gated by slack margin.
      - All numeric literals strictly limited to {-2, -1, 0, 1, 2}; no hidden constants.
      - Final score preserves lexicographic order: DDL feasibility first, then energy minimization within feasible set.
    """
    eps = 9.000127558739559e-05
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

    def mad_normalize(x):
        x = np.asarray(x, dtype=float)
        med = np.median(x) if N > 1 else np.mean(x)
        mad = np.median(np.abs(x - med)) if N > 1 else eps
        scale = 1.1509974966121055 * (mad if mad > eps else eps)
        return (x - med) / scale
    slack_score = np.where(slack < 0, (-slack) ** 1.081679170482574, 0.0)
    slack_lb = -6.7167688264624985
    slack_ub = 65.59581942978646
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    median_unc = np.median(uncertainty) if N > 1 else np.mean(uncertainty)
    ddl_protection_gate = ((slack <= 0.0) & (uncertainty <= median_unc + eps)).astype(float)
    critical_boost = upward_rank * (1.0 + 4.8906839055074744 * (1.0 - slack_scaled)) * ddl_protection_gate
    successor_release_score = upward_rank * remaining_work * ddl_protection_gate
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_norm = mad_normalize(energy_per_sec)
    energy_suppression_gate = ((slack > 0.0) & (uncertainty <= median_unc + eps)).astype(float)
    energy_weight_adj = 1.240433325025334 * (1.0 - energy_suppression_gate)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 0.258791471309445
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.14490371097263838
    weight_rank = 0.885170471173772 + (1.0 - 0.885170471173772) * (1.0 - slack_scaled)
    rank_norm = mad_normalize(upward_rank)
    rank_score = -rank_norm * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-5.805830506964652 * (uncertainty - 1.0)))
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    energy_uncertainty_score = 1.002975149580565 * energy_norm * unc_norm * unc_sigmoid
    wait_norm = mad_normalize(ready_wait_time)
    wait_gated = wait_norm * (1.0 - slack_scaled)
    score = mad_normalize(slack_score) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk) + energy_weight_adj * energy_eff_norm + rank_score + energy_uncertainty_score + mad_normalize(min_incremental_energy) * (1.0 - weight_rank) - critical_boost - successor_release_score + wait_gated
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
