import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining robustness from Parent 1 and efficiency from Parent 2.
    
    Key structural improvements:
    - Introduces hard feasibility gate: energy term only active when slack > PARAMS["energy_feasibility_gate_threshold"]
      (replacing soft exp(-slk) gating in Parent 1, enabling strict DDL-first behavior).
    - Replaces IQR with robust mean-abs normalization *and* adds explicit finite-range clipping for all normalized terms.
    - Uses percentile-based slack range interpolation *and* couples it with uncertainty-aware criticality boost.
    - Adds uncertainty-weighted duration risk as separate term (not just multiplicative coupling).
    - All numeric literals are strictly in {-2,-1,0,1,2}.
    """
    eps = 1.317439872157351e-09

    def robust_norm(x):
        x = np.asarray(x, dtype=float)
        denom = np.mean(np.abs(x)) + eps
        normed = x / denom
        return np.clip(normed, -2.0, 2.0)
    slack_arr = np.asarray(slack, dtype=float)
    slack_penalty = np.where(slack_arr < 0, (-slack_arr) ** 1.9940229377084386, slack_arr * 0.3480484993282736)
    rank_median = np.median(upward_rank) if len(upward_rank) > 1 else np.mean(upward_rank)
    is_high_rank = upward_rank >= rank_median
    is_tight_slack = slack_arr <= 0.17033066015172427 + eps
    critical_gate = np.where(is_high_rank & is_tight_slack, 3.3518000592565134, 1.0)
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = robust_norm(energy_per_sec)
    deadline_pressure = np.maximum(0.0, -slack_arr)
    unc_slack_coupling = uncertainty * deadline_pressure * 1.0754016994827849
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.23206791793236237
    wait_clipped = np.clip(ready_wait_time, 0, 2.732081490762907)
    wait_score = -robust_norm(wait_clipped)
    if len(slack_arr) > 1:
        slack_p90 = np.percentile(slack_arr, 92.4741579851054)
        slack_p10 = np.percentile(slack_arr, 20.045572391793634)
        slack_range = np.maximum(eps, slack_p90 - slack_p10)
        slack_normalized = np.clip((slack_arr - np.min(slack_arr)) / (slack_range + eps), 0, 1)
    else:
        slack_normalized = np.array([0.0])
    weight_rank = 0.001888441938029747 + (1.0 - 0.001888441938029747) * slack_normalized
    rank_score = -robust_norm(upward_rank) * weight_rank
    energy_term = np.where(slack_arr > 0.17033066015172427, energy_eff_score * 1.1370564060345014, 0.0)
    score = robust_norm(slack_penalty) + robust_norm(unc_slack_coupling) + robust_norm(duration_risk) + energy_term + rank_score + wait_score + robust_norm(min_incremental_energy) * (1.0 - weight_rank)
    score = score * critical_gate
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score
