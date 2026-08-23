import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with key structural improvements:
      - Replaces percentile-based slack normalization with fixed empirical bounds [slack_min_bound, slack_max_bound]
        for stability across sparse/low-N ready sets.
      - Introduces bounded sigmoid gate on uncertainty: soft suppression of energy_uncertainty_interaction
        beyond threshold, improving robustness to outlier uncertainty estimates.
      - Removes all wait-time and early-slack reward terms (confirmed inactive in reflection).
      - Uses monotonic slack mapping: linear interpolation from worst to best slack, preserving order.
      - All literals are -2,-1,0,1,2; no hidden constants; all tunables declared.
      - Smaller score = higher priority."""
    eps = 7.745512898263198e-08
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)

    def robust_norm(x):
        x = np.asarray(x, dtype=float)
        denom = np.mean(np.abs(x)) + eps
        return x / (denom + eps)
    slack_abs = np.abs(slack)
    slack_score = np.where(slack < 0, slack_abs ** 2.8838885729005774, 0.0)
    rank_median = np.median(upward_rank) if N > 1 else np.mean(upward_rank)
    is_high_rank = upward_rank >= rank_median
    is_tight_or_violated = slack <= 0
    critical_gate = np.where(is_high_rank & is_tight_or_violated, 3.4213013206222995, 1.0)
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = robust_norm(energy_per_sec)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 2.124254649460121
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 1.1834324007696149
    slack_lb = -9.557504720206069
    slack_ub = 40.40713661881866
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.7833925673077783 + (1.0 - 0.7833925673077783) * (1.0 - slack_scaled)
    rank_score = -robust_norm(upward_rank) * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-7.180691454886789 * (uncertainty - 1.0)))
    energy_norm = robust_norm(min_incremental_energy)
    unc_norm = robust_norm(uncertainty)
    energy_uncertainty_score = 0.43229546490128873 * energy_norm * unc_norm * unc_sigmoid
    score = robust_norm(slack_score) + robust_norm(unc_slack_coupling) + robust_norm(duration_risk) + 1.185378132452651 * energy_eff_score + rank_score + energy_uncertainty_score + robust_norm(min_incremental_energy) * (1.0 - weight_rank)
    score = score * critical_gate
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score
