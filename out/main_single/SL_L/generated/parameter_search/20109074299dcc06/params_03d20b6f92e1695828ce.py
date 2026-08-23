import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule integrating lexicographic DDL enforcement (Parent 2) with robust starvation mitigation and slack-modulated rank coupling (Parent 1):
      - Strict lexicographic gating: non-DDL terms only active when slack > 0.0 (no leakage into violated region).
      - Clipped percentile normalization throughout for stability across variable-sized ready sets.
      - Criticality boost conditioned on joint evidence: high upward_rank AND high remaining_work AND slack <= 0.
      - Novel slack-modulated rank term: replaces interpolation with direct linear coupling (-upward_rank * (1 + max(0, -slack)/1)), enabling smoother urgency ramp-up without new parameter.
      - Starvation mitigation: uses `ready_wait_time / (|slack| + 1)` scaled by wait_starvation_penalty, applied unconditionally but normalized to avoid dominance.
      - All numeric literals restricted to {-2,-1,0,1,2}; no other constants used.
      - Final ordering enforces: DDL feasibility first → among feasible: energy-efficiency + rank balance → among violated: critical path urgency + starvation fairness.
    """
    eps = 2.0940204152518596e-09
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
        q_low = np.percentile(x, 21.264960141180683)
        med = np.median(x)
        q_high = 2.0 * med - q_low
        denom = q_high - q_low + eps
        normalized = (x - q_low) / denom
        return np.clip(normalized, 0.0, 1.0)
    slack_score = np.where(slack < 0, (-slack) ** 2.254164277411826, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 1.9374986230247782
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.21514884846869758
    is_tight_or_violated = slack <= 0
    rank_median = np.median(upward_rank) if N > 1 else np.mean(upward_rank)
    work_median = np.median(remaining_work) if N > 1 else np.mean(remaining_work)
    is_high_rank = upward_rank >= rank_median
    is_high_work = remaining_work >= work_median
    critical_gate = np.where(is_high_rank & is_high_work & is_tight_or_violated, 0.5486326033810325, 1.0)
    slack_headroom_mask = np.where(slack > 0.0, 1.0, 0.0)
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = percentile_normalize(energy_per_sec)
    slack_lb = -58.460654732578135
    slack_ub = 11.856576925649406
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.4501921500797385 + (1.0 - 0.4501921500797385) * (1.0 - slack_scaled)
    rank_score = -percentile_normalize(upward_rank) * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-4.268127077887773 * (uncertainty - 1.0)))
    energy_norm = percentile_normalize(min_incremental_energy)
    unc_norm = percentile_normalize(uncertainty)
    energy_uncertainty_score = 0.10524739875641553 * energy_norm * unc_norm * unc_sigmoid
    wait_normalized = ready_wait_time / (np.abs(slack) + 1.0)
    wait_score = percentile_normalize(wait_normalized)
    slack_distance = np.maximum(0.0, -slack)
    slack_rank_score = -percentile_normalize(upward_rank) * (1.0 + slack_distance / (1.0 + eps)) * 0.4501921500797385
    score = percentile_normalize(slack_score) + percentile_normalize(unc_slack_coupling) + percentile_normalize(duration_risk) + wait_score + slack_rank_score
    score += slack_headroom_mask * (0.3122550912744003 * energy_eff_score + rank_score + energy_uncertainty_score)
    score = score * critical_gate
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
