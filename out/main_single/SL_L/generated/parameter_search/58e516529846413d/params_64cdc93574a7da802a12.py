import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's stability and Parent 1's successor-aware leverage.
    
    Key structural improvements:
    - Retains Parent 2's fixed empirical slack bounds and bounded sigmoid uncertainty gate for robustness.
    - Integrates Parent 1's *conditional critical path leverage* gated by both slack>=0 AND uncertainty<=median_unc,
      but re-purposed as a *bonus term* (not multiplier) to avoid over-amplification while preserving DDL protection.
    - Replaces Parent 2's linear rank_slack_balance interpolation with *piecewise linear mapping*: 
      prioritizes rank only in the critical zone [slack_min_bound, 0], zero outside — sharpens deadline enforcement.
    - Uses robust mean-absolute normalization (with PARAMS["robustness_mad_factor"]) instead of simple mean-abs scaling
      for better outlier resistance and cross-seed stability.
    - Removes all wait-time terms (confirmed inactive) and replaces energy fallback with direct normalized energy
      weighted by slack pressure — cleaner and more interpretable.
    """
    eps = 2.5659673104137983e-06
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

    def robust_normalize(x):
        x = np.asarray(x, dtype=float)
        mu = np.mean(x)
        mad = np.mean(np.abs(x - mu))
        scale = 1.0042696072635933 * (mad if mad > eps else eps)
        return (x - mu) / (scale + eps)
    slack_score = np.where(slack < 0, (-slack) ** 2.641081209404276, 0.0)
    duration_total = min_exec_time + min_comm_time + eps
    duration_norm = robust_normalize(duration_total)
    duration_risk = duration_norm * uncertainty * 0.8229812465621045
    energy_per_sec = min_incremental_energy / (duration_total + eps)
    energy_eff_score = robust_normalize(energy_per_sec) * 0.8607719683555085
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = robust_normalize(uncertainty * deadline_pressure) * 0.6903767822714305
    median_unc = np.median(uncertainty)
    ddl_protection_gate = ((slack >= 0.0) & (uncertainty <= median_unc + eps)).astype(float)
    rank_norm = robust_normalize(upward_rank)
    critical_bonus = 2.2583143163633523 * rank_norm * ddl_protection_gate
    slack_lb = -0.21929769365075913
    slack_ub = 0.0
    in_critical_zone = (slack >= slack_lb) & (slack <= slack_ub)
    rank_weight = np.where(in_critical_zone, 0.8547524517003011 * (1.0 - (slack - slack_lb) / (slack_ub - slack_lb + eps)), 0.0)
    rank_score = -robust_normalize(upward_rank) * rank_weight
    unc_sigmoid = 1.0 / (1.0 + np.exp(-1.3545364156604178 * (uncertainty - 1.0)))
    energy_norm = robust_normalize(min_incremental_energy)
    unc_norm = robust_normalize(uncertainty)
    energy_uncertainty_score = 1.2982340832660428 * energy_norm * unc_norm * unc_sigmoid
    slack_pressure = np.maximum(0.0, -slack)
    energy_base_weight = np.exp(-slack_pressure * 0.17419183758571105)
    energy_base_score = robust_normalize(min_incremental_energy) * energy_base_weight
    score = robust_normalize(slack_score) + duration_risk + energy_eff_score + unc_slack_coupling + energy_uncertainty_score + energy_base_score - critical_bonus - rank_score
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
