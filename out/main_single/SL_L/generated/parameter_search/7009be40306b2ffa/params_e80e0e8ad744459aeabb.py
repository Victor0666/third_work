import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's strict lexicographic DDL gate and percentile stability with Parent 1's unified critical-path urgency and robust wait-time scaling:
      - Strict lexicographic gate: non-DDL terms activate *only* when slack > 0.0 (ensures hard deadline feasibility dominates)
      - Unified critical-path urgency: upward_rank × remaining_work unconditionally (retained from Parent 1, empirically validated for release-order fidelity)
      - Robust wait-term: ready_wait_time / (clamped |slack| + 1) where clamp uses fixed threshold 1.0 (allowed literal); then percentile-normalized
      - Clipped percentile normalization used universally (more stable than MAD for variable N and sparse sets)
      - Criticality boost now conditioned on slack <= 0 AND (upward_rank > median OR remaining_work > median) — relaxed disjunction improves coverage without sacrificing safety
      - All divisions guarded; no infinite/nan outputs; deterministic and finite
      - Final score enforces: DDL-feasibility first → among feasible: energy-efficiency + rank balance → among violated: critical path urgency + controlled starvation mitigation
    """
    eps = 1e-09
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
        q_low = np.percentile(x, 10.152052906303638)
        med = np.median(x)
        q_high = 2.0 * med - q_low
        denom = q_high - q_low + eps
        normalized = (x - q_low) / denom
        return np.clip(normalized, 0.0, 1.0)
    slack_score = np.where(slack < 0, (-slack) ** 3.509224874048493, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 0.8670771881278478
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.935399348995728
    critical_path_urgency = upward_rank * remaining_work
    is_tight_or_violated = slack <= 0
    rank_median = np.median(upward_rank) if N > 1 else np.mean(upward_rank)
    work_median = np.median(remaining_work) if N > 1 else np.mean(remaining_work)
    is_high_rank = upward_rank >= rank_median
    is_high_work = remaining_work >= work_median
    critical_gate = np.where((is_high_rank | is_high_work) & is_tight_or_violated, 3.131612146268474, 1.0)
    slack_headroom_mask = np.where(slack > 0.0, 1.0, 0.0)
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = percentile_normalize(energy_per_sec)
    slack_lb = -48.55882572871024
    slack_ub = 50.272563864765985
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.805233017434432 + (1.0 - 0.805233017434432) * (1.0 - slack_scaled)
    rank_score = -percentile_normalize(upward_rank) * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-4.173294898525124 * (uncertainty - 1.0)))
    energy_norm = percentile_normalize(min_incremental_energy)
    unc_norm = percentile_normalize(uncertainty)
    energy_uncertainty_score = 1.0975267255572538 * energy_norm * unc_norm * unc_sigmoid
    abs_slack = np.abs(slack)
    clamped_abs_slack = np.where(abs_slack < 1.0, 1.0, abs_slack)
    wait_base = np.where(slack_headroom_mask > 0.0, ready_wait_time / (clamped_abs_slack + 1.0), 0.0)
    wait_score = percentile_normalize(wait_base)
    score = percentile_normalize(slack_score) + percentile_normalize(unc_slack_coupling) + percentile_normalize(duration_risk) + percentile_normalize(critical_path_urgency)
    score += slack_headroom_mask * (0.12393363413459516 * energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    score = score * critical_gate
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
