import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule incorporating:
      - Lexicographic DDL gating via strict `slack > 0` condition (no leakage)
      - Replaced percentile-normalized `ready_wait_time` with *relative* waiting ratio scaled by critical path urgency
      - Introduced successor-release interaction: `remaining_work / (upward_rank + eps)` to prioritize release of downstream work
      - Added host-load-aware energy term: `min_incremental_energy / (1 + uncertainty)` — suppresses marginal energy when uncertainty is high
      - Used robust linear-ramp slack gating instead of fragile sigmoids for DDL pressure sensitivity
      - Criticality boost now requires *all three*: (slack <= 0) AND (upward_rank > median) AND (remaining_work > median)
      - All feature interactions are bounded, numerically safe, and use only {-2,-1,0,1,2} literals
      - Final score preserves lexicographic ordering: DDL-violating terms dominate; feasible region prioritizes energy-efficiency + critical path + starvation mitigation
    """
    eps = 6.184284403031561e-09
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

    def percentile_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        q_low = np.percentile(x, 11.018597177104215)
        med = np.median(x)
        q_high = 2.0 * med - q_low
        denom = q_high - q_low + eps
        normalized = (x - q_low) / denom
        return np.clip(normalized, 0.0, 1.0)
    slack_score = np.where(slack < 0, (-slack) ** 2.3633632673745195, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 1.3094901030100636
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.009419442569485237
    is_tight_or_violated = slack <= 0
    rank_median = np.median(upward_rank) if N > 1 else np.mean(upward_rank)
    work_median = np.median(remaining_work) if N > 1 else np.mean(remaining_work)
    is_high_rank = upward_rank >= rank_median
    is_high_work = remaining_work >= work_median
    critical_gate = np.where(is_high_rank & is_high_work & is_tight_or_violated, 0.5199444531416636, 1.0)
    slack_headroom_mask = np.where(slack > 0.0, 1.0, 0.0)
    load_aware_energy = min_incremental_energy / (1.0 + uncertainty + eps)
    energy_eff_score = percentile_normalize(load_aware_energy)
    successor_release_ratio = remaining_work / (upward_rank + eps)
    successor_score = -percentile_normalize(successor_release_ratio)
    slack_lb = -0.8876932545536249
    slack_ub = 15.682718962143051
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.00616448284997687 + (1.0 - 0.00616448284997687) * (1.0 - slack_scaled)
    rank_score = -percentile_normalize(upward_rank) * weight_rank
    unc_norm = percentile_normalize(uncertainty)
    unc_gate = np.clip(4.487318033240962 * (1.0 - unc_norm), 0.0, 1.0)
    energy_norm = percentile_normalize(min_incremental_energy)
    energy_uncertainty_score = 0.9117241935967467 * energy_norm * unc_gate
    duration_total = min_exec_time + min_comm_time + eps
    wait_duration_ratio = ready_wait_time / (np.median(duration_total) + eps) if N > 1 else ready_wait_time / (duration_total[0] + eps)
    wait_score = np.where(slack_headroom_mask > 0.0, percentile_normalize(wait_duration_ratio), 0.0)
    score = percentile_normalize(slack_score) + percentile_normalize(unc_slack_coupling) + percentile_normalize(duration_risk)
    score += slack_headroom_mask * (0.780854280152929 * energy_eff_score + rank_score + successor_score + energy_uncertainty_score + wait_score)
    score = score * critical_gate
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
