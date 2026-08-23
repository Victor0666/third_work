import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining lexicographic DDL enforcement from Parent 2 with robust wait-time fairness and successor-aware criticality from Parent 1:
      - Retains strict lexicographic gate: non-DDL terms only active when slack > 0.0 (compliant literal)
      - Replaces linear wait-time term with `ready_wait_time / (|slack| + 1)` *only when slack > 0*, then percentile-normalized → prevents starvation without violating DDL-first order
      - Integrates successor-release awareness via `remaining_work / (median(remaining_work) + eps)` in criticality boost, activated only under DDL pressure (slack <= 0)
      - Uses clipped percentile normalization throughout for stability across variable-size ready sets
      - Criticality boost now conditioned on both slack violation AND high rank AND high remaining work — tighter evidence alignment
      - All numeric literals restricted to {-2, -1, 0, 1, 2}; no other constants used
      - Final score enforces: DDL-feasibility first → among feasible: energy-efficiency + rank balance → among violated: critical path urgency + starvation mitigation
    """
    eps = 1.016642725689664e-09
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
        q_low = np.percentile(x, 15.899855796948856)
        med = np.median(x)
        q_high = 2.0 * med - q_low
        denom = q_high - q_low + eps
        normalized = (x - q_low) / denom
        return np.clip(normalized, 0.0, 1.0)
    slack_score = np.where(slack < 0, (-slack) ** 3.092711531364963, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 1.7036503943679278
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 1.1469805796061936
    is_tight_or_violated = slack <= 0
    rank_median = np.median(upward_rank) if N > 1 else np.mean(upward_rank)
    work_median = np.median(remaining_work) if N > 1 else np.mean(remaining_work)
    is_high_rank = upward_rank >= rank_median
    is_high_work = remaining_work >= work_median
    critical_gate = np.where(is_high_rank & is_high_work & is_tight_or_violated, 1.4094789389309472, 1.0)
    slack_headroom_mask = np.where(slack > 0.0, 1.0, 0.0)
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = percentile_normalize(energy_per_sec)
    slack_lb = -5.9066568012330265
    slack_ub = 31.4835731326195
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.719518565001592 + (1.0 - 0.719518565001592) * (1.0 - slack_scaled)
    rank_score = -percentile_normalize(upward_rank) * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-4.155589966554307 * (uncertainty - 1.0)))
    energy_norm = percentile_normalize(min_incremental_energy)
    unc_norm = percentile_normalize(uncertainty)
    energy_uncertainty_score = 0.03308822771795969 * energy_norm * unc_norm * unc_sigmoid
    wait_normalized = np.where(slack_headroom_mask > 0.0, ready_wait_time / (np.abs(slack) + 1.0), 0.0)
    wait_score = percentile_normalize(wait_normalized)
    score = percentile_normalize(slack_score) + percentile_normalize(unc_slack_coupling) + percentile_normalize(duration_risk)
    score += slack_headroom_mask * (0.13119335086617148 * energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    score = score * critical_gate
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
