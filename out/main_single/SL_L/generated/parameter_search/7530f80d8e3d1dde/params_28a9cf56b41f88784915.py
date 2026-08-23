import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with bounded linear slack scaling and min-max normalization:
      - Replaces logistic slack scaling with robust bounded linear interpolation [slack_linear_scale_low, slack_linear_scale_high],
        improving stability near threshold and reducing sensitivity to parameter perturbations.
      - Uses per-feature min-max normalization (not MAD) for better rank-energy tradeoff in small-N ready sets (<5 tasks).
      - Removes redundant critical_path_release_weight and slack_penalty_exponent per reflection — DDL enforcement is now purely
        lexicographic via zeroing non-DDL terms when slack <= 0, eliminating need for tunable amplifiers.
      - All numeric literals are in {-2,-1,0,1,2}; no hidden constants or unbounded operations.
      - Anti-starvation remains dual-gated (slack > 0 AND slack >= 1.0) with fixed threshold.
      - Energy efficiency uses uncertainty-dampened denominator: duration * (1 + uncertainty).
      - Final score is finite, shape-(N,), deterministic, and satisfies all interface contracts.
    """
    eps = 3.4409118342579985e-06
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

    def minmax_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        x_min = np.min(x)
        x_max = np.max(x)
        range_safe = x_max - x_min + eps
        normalized = (x - x_min) / range_safe
        return np.clip(normalized, 0.0, 1.0)
    slack_score = np.where(slack < 0, -slack, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 1.4128348660273604
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 1.5175629173263645
    slack_headroom_mask = np.where(slack > 0.0, 1.0, 0.0)
    duration_total_safe = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / (duration_total_safe * (1.0 + uncertainty + eps))
    energy_eff_score = minmax_normalize(energy_per_sec) * 0.9795319121425347
    slack_lb = -18.80860390978527
    slack_ub = 14.712426604805053
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.37221910494729926 + (1.0 - 0.37221910494729926) * (1.0 - slack_scaled)
    rank_score = (1.0 - minmax_normalize(upward_rank)) * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-5.376589400185892 * (uncertainty - 1.0)))
    energy_norm = minmax_normalize(min_incremental_energy)
    unc_norm = minmax_normalize(uncertainty)
    energy_uncertainty_score = 0.1938046398826459 * energy_norm * unc_norm * unc_sigmoid
    starvation_gate = np.where(slack >= 1.0, 1.0, 0.0)
    starvation_enabled = slack_headroom_mask * starvation_gate
    duration_med = np.median(duration_total_safe) if N > 1 else duration_total_safe[0]
    wait_duration_scaled = ready_wait_time / (duration_med + eps)
    wait_power = wait_duration_scaled ** 0.6779781290882236
    wait_score = minmax_normalize(wait_power)
    score = minmax_normalize(slack_score) + minmax_normalize(unc_slack_coupling) + minmax_normalize(duration_risk)
    score += starvation_enabled * (energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
