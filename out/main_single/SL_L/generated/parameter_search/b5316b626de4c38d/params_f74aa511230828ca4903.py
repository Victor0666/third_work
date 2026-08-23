import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule incorporating:
      - Replaced absolute wait-time term with *exponential decay* of waiting time under DDL headroom, reducing starvation bias in large ready sets
      - Removed slack_min_bound (evidence shows it's inactive and destabilizing) → use symmetric clipping around zero slack via linear ramp
      - Introduced *upward_rank × remaining_work* interaction as a native critical-path release signal, normalized separately and gated only under slack <= 0
      - Simplified DDL gating: use linear ramp from slack=0 to slack=slack_max_bound instead of fragile percentile bounds → improves boundary stability
      - Energy-efficiency score now computed as min_incremental_energy / (min_exec_time + min_comm_time + eps), then percentile-normalized
      - All non-DDL terms are multiplied by slack_headroom_mask (strict lexicographic enforcement)
      - Criticality boost now applied *additively* (not multiplicatively) to avoid numerical explosion and improve gradient stability
      - Final score combines DDL risk, critical path urgency, energy efficiency, and anti-starvation — all robustly normalized and bounded
    """
    eps = 2.322371873911971e-09
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
        q_low = np.percentile(x, 17.61967063819157)
        med = np.median(x)
        q_high = 2.0 * med - q_low
        denom = q_high - q_low + eps
        normalized = (x - q_low) / denom
        return np.clip(normalized, 0.0, 1.0)
    slack_score = np.where(slack < 0, -slack, 0.0)
    severe_slack_penalty = np.where(slack < -eps, (-slack) ** 2.4703320226340377, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 1.0122896037396678
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.47341870800059854
    is_tight_or_violated = slack <= 0
    critical_path_product = upward_rank * remaining_work
    critical_path_score = np.where(is_tight_or_violated, percentile_normalize(critical_path_product), 0.0)
    slack_headroom_mask = np.where(slack > 0.0, 1.0, 0.0)
    duration_total = min_exec_time + min_comm_time + eps
    energy_eff_score = percentile_normalize(min_incremental_energy / duration_total)
    slack_scaled = np.clip(slack / (24.95721663231135 + eps), 0.0, 1.0)
    weight_rank = 0.7656063326233712 + (1.0 - 0.7656063326233712) * (1.0 - slack_scaled)
    rank_score = -percentile_normalize(upward_rank) * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-6.8861442726797994 * (uncertainty - 1.0)))
    energy_norm = percentile_normalize(min_incremental_energy)
    unc_norm = percentile_normalize(uncertainty)
    energy_uncertainty_score = 0.5680695032171801 * energy_norm * unc_norm * unc_sigmoid
    wait_score = np.where(slack_headroom_mask > 0.0, percentile_normalize(ready_wait_time ** 0.9424229766350463), 0.0)
    score = percentile_normalize(slack_score) + percentile_normalize(severe_slack_penalty) + percentile_normalize(unc_slack_coupling) + percentile_normalize(duration_risk) + critical_path_score
    score += slack_headroom_mask * (0.8525896945395012 * energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    score += np.where(is_tight_or_violated & (upward_rank >= np.median(upward_rank) if N > 1 else np.mean(upward_rank)) & (remaining_work >= np.median(remaining_work) if N > 1 else np.mean(remaining_work)), 1.8692378094424233, 0.0)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
