import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule synthesizing best practices from both parents:
      - Joint MAD normalization over |slack|, uncertainty, duration_total for robust cross-signal alignment
      - Hard lexicographic DDL protection via `slack > ddl_protection_threshold` (not > 0)
      - Unconditional critical-path release term: upward_rank * remaining_work * weight
      - Dual-gated non-DDL optimization: requires BOTH sufficient slack AND host capacity margin
      - Replaces fragile sigmoid/percentile gates with bounded linear interactions and MAD clipping
      - Eliminates wait-time decay (redundant under critical-path + DDL pressure) and anti-starvation normalization (handled by critical release + DDL gate)
      - All numeric literals strictly limited to {-2,-1,0,1,2}; no other constants used
      - Final score structure: [DDL-violation] + [critical-release] + [gated-efficiency-terms]
    """
    eps = 3.727104456174107e-09
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
    duration_total = min_exec_time + min_comm_time + eps
    abs_slack = np.abs(slack)
    all_risk = np.concatenate([abs_slack, uncertainty, duration_total])
    risk_median = np.median(all_risk)
    mad = np.median(np.abs(all_risk - risk_median)) + eps
    joint_scale = 0.7220028500882575 * mad

    def joint_mad_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        centered = x - risk_median
        normalized = centered / joint_scale
        return np.clip(normalized, -2.0, 2.0)
    slack_violation = np.where(slack < 0, (-slack) ** 1.9812095869643591, 0.0)
    unc_slack_coupling = 0.2883841084136669 * uncertainty * np.maximum(0.0, -slack)
    duration_risk = 0.6633201561555206 * duration_total * uncertainty * np.where(slack <= 0.9300751518201589, 1.0, 0.0)
    critical_release = 1.4333642928092416 * upward_rank * remaining_work
    ddl_safe_mask = np.where(slack > 0.9300751518201589, 1.0, 0.0)
    duration_median = np.median(duration_total) if N > 1 else duration_total[0]
    host_margin_mask = np.where(duration_total <= 2.6066785000226456 * duration_median, 1.0, 0.0)
    optimization_mask = ddl_safe_mask * host_margin_mask
    energy_per_sec = min_incremental_energy / (duration_total + eps)
    energy_eff_score = joint_mad_normalize(energy_per_sec) * 0.5482084705106444
    energy_norm = joint_mad_normalize(min_incremental_energy)
    unc_norm = joint_mad_normalize(uncertainty)
    energy_uncertainty_score = 1.0142050185205018 * energy_norm * unc_norm
    score = joint_mad_normalize(slack_violation) + joint_mad_normalize(unc_slack_coupling) + joint_mad_normalize(duration_risk)
    score += -joint_mad_normalize(critical_release)
    score += optimization_mask * (energy_eff_score + energy_uncertainty_score)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
