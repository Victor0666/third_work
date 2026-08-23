import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's stability and Parent 1's robust normalization:
      - Retains fixed empirical slack bounds [slack_min_bound, slack_max_bound] for sparse-set stability.
      - Replaces mean-abs normalization with MAD-based robust_norm using PARAMS["robustness_mad_factor"].
      - Introduces critical path leverage gated by *both* slack <= 0 AND low uncertainty (from Parent 1),
        preventing premature boosting of high-rank but high-risk tasks.
      - Keeps bounded sigmoid uncertainty gate for energy-uncertainty interaction.
      - Drops all wait-time and early-slack reward terms (confirmed inactive).
      - Uses piecewise slack scoring: power penalty for negative, zero for non-negative (hard deadline focus).
      - All operations are bounded, epsilon-guarded, deterministic, and shape-preserving."""
    eps = 1.1210871472832559e-07
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
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        scale = 1.4263250437186274 * (mad if mad > eps else eps)
        return (x - med) / (scale + eps)
    slack_score = np.where(slack < 0, (-slack) ** 2.368395609552171, 0.0)
    unc_med = np.median(uncertainty) if N > 1 else np.mean(uncertainty)
    unc_gate = np.clip((2.0 * unc_med - uncertainty) / (unc_med + eps), 0.0, 1.0)
    is_tight_or_violated = slack <= 0
    rank_median = np.median(upward_rank) if N > 1 else np.mean(upward_rank)
    is_high_rank = upward_rank >= rank_median
    critical_gate = np.where(is_high_rank & is_tight_or_violated & (unc_gate > 0.0), 1.6548755200838106, 1.0)
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = mad_normalize(energy_per_sec)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 0.0
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.9013192573902296
    slack_lb = -5.140321349204754
    slack_ub = 1.8308769853678237
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.4623247519394471 + (1.0 - 0.4623247519394471) * (1.0 - slack_scaled)
    rank_score = -mad_normalize(upward_rank) * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-4.651609708907463 * (uncertainty - 1.0)))
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    energy_uncertainty_score = 0.48825453168890376 * energy_norm * unc_norm * unc_sigmoid
    score = mad_normalize(slack_score) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk) + 1.484058845177873 * energy_eff_score + rank_score + energy_uncertainty_score + mad_normalize(min_incremental_energy) * (1.0 - weight_rank)
    score = score * critical_gate
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
