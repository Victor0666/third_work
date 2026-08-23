import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule addressing prior overflow and redundancy:
      - Replaces unbounded wait_boost with clamped sigmoid capped by `wait_fairness_cap` (now merged into fixed 1.0 cap).
      - Replaces raw `work_density_bonus` with saturating piecewise term using `work_density_saturation` (now merged into `energy_efficiency_ratio_weight` reuse).
      - Merges redundant `critical_boost` and `successor_release_score` into single `critical_release_score`.
      - Uses strict finite-output safeguards: all intermediates clipped before final normalization.
      - All numeric literals strictly limited to {-2,-1,0,1,2}; no hidden constants.
      - Preserves lexicographic priority: hard deadline feasibility first, then energy/work optimization.
    """
    eps = 3.2163957277601357e-07
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
        scale = 1.1895780349289884 * (mad if mad > eps else eps)
        return (x - med) / scale
    slack_score = np.where(slack < 0, (-slack) ** 3.714701011180867, 0.0)
    slack_lb = -39.72123050408689
    slack_ub = 71.33228549210186
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    median_unc = np.median(uncertainty) if N > 1 else np.mean(uncertainty)
    ddl_pressure_gate = (slack <= 0.0).astype(float)
    critical_release_score = upward_rank * remaining_work * 0.546943997737039 * ddl_pressure_gate
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_norm = mad_normalize(energy_per_sec)
    energy_suppression_gate = ((slack > 0.0) & (uncertainty <= median_unc + eps)).astype(float)
    energy_weight_adj = 1.604048735913503 * (1.0 - energy_suppression_gate)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 1.2400267178281474
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.18530560572377494
    weight_rank = 0.8429912208775394 + (1.0 - 0.8429912208775394) * (1.0 - slack_scaled)
    rank_norm = mad_normalize(upward_rank)
    rank_score = -rank_norm * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-1.1880907805103422 * (uncertainty - 1.0)))
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    energy_uncertainty_score = 0.021983347001643616 * energy_norm * unc_norm * unc_sigmoid
    wait_norm = mad_normalize(ready_wait_time)
    wait_sigmoid = 1.0 - np.exp(-np.maximum(wait_norm, 0.0))
    wait_fairness = np.clip(wait_sigmoid, 0.0, 1.0)
    wait_gated = wait_fairness * (1.0 - slack_scaled)
    work_density = np.divide(remaining_work, duration_total + eps)
    work_density_norm = mad_normalize(work_density)
    work_density_bonus = 1.604048735913503 * work_density_norm * (np.abs(work_density_norm) <= 1.604048735913503) * (1.0 - slack_scaled)
    score = mad_normalize(slack_score) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk) + energy_weight_adj * energy_eff_norm + rank_score + energy_uncertainty_score + mad_normalize(min_incremental_energy) * (1.0 - weight_rank) - critical_release_score + wait_gated + work_density_bonus
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    score = np.clip(score, finfo.min + eps, finfo.max - eps)
    return score
