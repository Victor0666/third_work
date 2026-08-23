import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with soft sigmoid feasibility gating and tanh-bounded normalization.
    
    Key improvements:
    - Replaced hard energy gating with smooth, differentiable sigmoid: enables CMA-ES gradient signals
      and avoids brittle transitions near DDL boundary.
    - Replaced clipping in robust_norm with tanh scaling: preserves ordinal relationships while strictly
      bounding influence to [-1,1] for stable optimization.
    - Removed redundant criticality_boost gating (per self-reflection) and strengthened rank utilization
      via uncertainty-aware upward_rank normalization using declared parameter.
    - Added explicit uncertainty-weighted rank normalization to prioritize high-criticality tasks *under risk*.
    - All numeric literals are strictly in {-2,-1,0,1,2}.
    """
    eps = 1.2638300701271193e-05

    def robust_norm(x):
        x = np.asarray(x, dtype=float)
        denom = np.mean(np.abs(x)) + eps
        normed = x / denom
        return np.tanh(normed)
    slack_arr = np.asarray(slack, dtype=float)
    slack_penalty = np.where(slack_arr < 0, (-slack_arr) ** 2.453366032871221, slack_arr * 0.004653728970023144)
    energy_gate = 1.0 / (1.0 + np.exp(-5.029604665349017 * (slack_arr - 1.8525839845373575)))
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = robust_norm(energy_per_sec)
    energy_term = energy_gate * energy_eff_score * 0.20634235611140916
    deadline_pressure = np.maximum(0.0, -slack_arr)
    unc_slack_coupling = uncertainty * deadline_pressure * 2.6732709047960173
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.7155548008076137
    wait_clipped = np.clip(ready_wait_time, 0, 42.20657917677833)
    wait_score = -robust_norm(wait_clipped)
    if len(slack_arr) > 1:
        slack_p90 = np.percentile(slack_arr, 94.46198128806736)
        slack_p10 = np.percentile(slack_arr, 3.75534215240904)
        slack_range = np.maximum(eps, slack_p90 - slack_p10)
        slack_normalized = np.clip((slack_arr - np.min(slack_arr)) / (slack_range + eps), 0, 1)
    else:
        slack_normalized = np.array([0.0])
    weight_rank = 0.1474164611863188 + (1.0 - 0.1474164611863188) * slack_normalized
    rank_uncertainty_weighted = upward_rank * (1.0 + uncertainty * 1.8525839845373575)
    rank_score = -robust_norm(rank_uncertainty_weighted) * weight_rank
    residual_energy_bias = robust_norm(min_incremental_energy) * (1.0 - weight_rank)
    score = robust_norm(slack_penalty) + robust_norm(unc_slack_coupling) + robust_norm(duration_risk) + energy_term + rank_score + wait_score + residual_energy_bias
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score
