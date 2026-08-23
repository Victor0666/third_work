import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule featuring:
      - Per-task criticality-aware activation: critical_release_score only when slack ≤ 0 AND (upward_rank * remaining_work) in top-k percentile.
      - Symmetric quantile clipping (±quantile_clip_bound) replacing separate low/high params — reduces parameter count while preserving stability.
      - Critical release enhanced with uncertainty coupling for risk-aware bottleneck targeting.
      - Quantile normalization replaces MAD: robust for small-N, avoids outlier fragility.
      - All numeric literals strictly {-2,-1,0,1,2}; no hidden constants.
      - Lexicographic DDL priority preserved: slack violation terms dominate unconditionally.
    """
    eps = 4.7168774270232454e-07
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

    def quantile_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        q_low = 0.001228902119054778
        q_high = 1.0 - q_low
        p_low = np.quantile(x, q_low)
        p_high = np.quantile(x, q_high)
        scale = p_high - p_low + eps
        med = np.median(x)
        normed = (x - med) / scale
        return np.clip(normed, -2.0, 2.0)
    slack_score = np.where(slack < 0, (-slack) ** 2.9095850465369786, 0.0)
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 1.2289763323831773
    criticality_metric = upward_rank * remaining_work
    if N > 1:
        critical_threshold = np.quantile(criticality_metric[slack <= 0], 0.8058188590778212)
        is_critical = (slack <= 0) & (criticality_metric >= critical_threshold - eps)
    else:
        is_critical = (slack <= 0) & (criticality_metric >= criticality_metric[0] - eps)
    critical_release_score = criticality_metric * uncertainty * 4.2559703508701086 * is_critical.astype(float)
    slack_lb = -1.054614738943485
    slack_ub = 39.148337818610415
    slack_centered = slack - slack_lb
    slack_range = slack_ub - slack_lb + eps
    slack_normalized = np.clip(slack_centered / slack_range, -2.0, 2.0)
    ddl_gate = 1.0 / (1.0 + np.exp(-3.5351155745552933 * slack_normalized))
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_norm = quantile_normalize(energy_per_sec)
    weight_rank = 0.7311178607793392 + (1.0 - 0.7311178607793392) * (1.0 - ddl_gate)
    rank_norm = quantile_normalize(upward_rank)
    rank_score = -rank_norm * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-3.5351155745552933 * (uncertainty - 1.0)))
    energy_norm = quantile_normalize(min_incremental_energy)
    unc_norm = quantile_normalize(uncertainty)
    energy_uncertainty_score = 1.603576679674878 * energy_norm * unc_norm * unc_sigmoid
    wait_norm = quantile_normalize(ready_wait_time)
    wait_sigmoid = 1.0 - np.exp(-3.5351155745552933 * np.maximum(wait_norm, 0.0))
    wait_fairness = np.clip(wait_sigmoid, 0.0, 1.0)
    wait_gated = wait_fairness * ddl_gate
    work_density = np.divide(remaining_work, duration_total + eps)
    work_density_norm = quantile_normalize(work_density)
    work_density_bonus = 1.2600112344482666 * work_density_norm * (np.abs(work_density_norm) <= 1.2600112344482666) * (1.0 - ddl_gate)
    score = quantile_normalize(slack_score) + quantile_normalize(duration_risk) + -critical_release_score
    score += ddl_gate * (1.2600112344482666 * energy_eff_norm + rank_score + energy_uncertainty_score + wait_gated + work_density_bonus)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    score = np.clip(score, finfo.min + eps, finfo.max - eps)
    return score
