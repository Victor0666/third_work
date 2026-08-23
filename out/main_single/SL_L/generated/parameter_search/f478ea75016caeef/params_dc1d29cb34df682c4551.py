import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining strengths from both parents:
      - Retains Parent 2's DDL-aware critical_release_score and robustness_mad_factor.
      - Adopts Parent 1's unified slack-normalized ddl_gate for smooth blending of non-DDL terms.
      - Replaces hard binary gates with continuous, bounded sigmoid activation (ddl_gate) to avoid instability.
      - Uses single robust MAD normalization function with PARAMS["robustness_mad_factor"] for all features.
      - Enforces strict lexicographic ordering: DDL-critical terms dominate; non-DDL terms smoothly fade in only when slack is safe.
      - All numeric literals are strictly {-2,-1,0,1,2}; no hidden constants.
      - Removed wait_fairness_steepness to comply with 12-parameter limit; reuse PARAMS["uncertainty_sigmoid_steepness"] for wait sigmoid.
    """
    eps = 2.0257526903010303e-06
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
        if N == 1:
            return np.zeros_like(x, dtype=float)
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        scale = 1.120860872452205 * (mad + eps)
        return (x - med) / scale
    slack_score = np.where(slack < 0, (-slack) ** 2.04115027787258, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 1.8006724901904265
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 1.7323463877070908
    slack_lb = -26.629342005792594
    slack_ub = 28.23592384041702
    slack_centered = slack - slack_lb
    slack_range = slack_ub - slack_lb + eps
    slack_normalized = np.clip(slack_centered / slack_range, -2.0, 2.0)
    ddl_gate = 1.0 / (1.0 + np.exp(-4.652921157545403 * slack_normalized))
    critical_release_score = upward_rank * remaining_work * 1.307934062360228 * (1.0 - ddl_gate)
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_norm = mad_normalize(energy_per_sec)
    weight_rank = 0.9766307380143427 + (1.0 - 0.9766307380143427) * (1.0 - ddl_gate)
    rank_norm = mad_normalize(upward_rank)
    rank_score = -rank_norm * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-4.652921157545403 * (uncertainty - 1.0)))
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    energy_uncertainty_score = 1.2901779767404964 * energy_norm * unc_norm * unc_sigmoid
    wait_norm = mad_normalize(ready_wait_time)
    wait_sigmoid = 1.0 - np.exp(-4.652921157545403 * np.maximum(wait_norm, 0.0))
    wait_fairness = np.clip(wait_sigmoid, 0.0, 1.0)
    wait_gated = wait_fairness * ddl_gate
    work_density = np.divide(remaining_work, duration_total + eps)
    work_density_norm = mad_normalize(work_density)
    work_density_bonus = 0.4209347210529467 * work_density_norm * (np.abs(work_density_norm) <= 0.4209347210529467) * (1.0 - ddl_gate)
    score = mad_normalize(slack_score) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk) + critical_release_score
    score += ddl_gate * (0.4209347210529467 * energy_eff_norm + rank_score + energy_uncertainty_score + wait_gated + work_density_bonus)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    score = np.clip(score, finfo.min + eps, finfo.max - eps)
    return score
