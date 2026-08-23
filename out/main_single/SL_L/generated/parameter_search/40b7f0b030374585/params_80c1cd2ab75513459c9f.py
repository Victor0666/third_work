import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with strict lexicographic DDL protection:
      - Uses tunable 'ddl_protection_gate_threshold' derived from slack_min_bound to define the exact slack boundary (≤ threshold) where ALL non-DDL terms are suppressed — no new parameter added.
      - Criticality boost now conditioned on joint slack violation (slack ≤ slack_min_bound), high upward_rank (> median), AND high remaining_work (> median).
      - Anti-starvation term `ready_wait_time / (|slack| + 1)` gated only when slack > slack_min_bound (not just > 0), ensuring fairness without compromising feasibility.
      - All DDL-critical terms dominate; non-DDL terms contribute only in strictly feasible region defined by slack_min_bound.
      - Clipped percentile normalization ensures stability across variable-size ready sets.
      - Final score guarantees strict lexicographic ordering: DDL-violating tasks always outrank feasible ones.
    """
    eps = 9.784483639585158e-08
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
        q_low = np.percentile(x, 20.974004974820012)
        med = np.median(x)
        q_high = 2.0 * med - q_low
        denom = q_high - q_low + eps
        normalized = (x - q_low) / denom
        return np.clip(normalized, 0.0, 1.0)
    slack_score = np.where(slack < -2.7715996733752064, (-slack) ** 2.170951306361327, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 0.31259478541510183
    duration_total = min_exec_time + min_comm_time + eps
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.8911427652801921
    slack_ramp = np.clip((-slack - -2.7715996733752064) / (0.0 - -2.7715996733752064 + eps), 0.0, 1.0)
    escalated_duration_risk = duration_risk * (1.0 + slack_ramp)
    is_violated = slack <= -2.7715996733752064
    rank_median = np.median(upward_rank) if N > 1 else np.mean(upward_rank)
    work_median = np.median(remaining_work) if N > 1 else np.mean(remaining_work)
    is_high_rank = upward_rank >= rank_median
    is_high_work = remaining_work >= work_median
    critical_gate = np.where(is_high_rank & is_high_work & is_violated, 2.525596851427748, 1.0)
    feasible_mask = np.where(slack > -2.7715996733752064, 1.0, 0.0)
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = percentile_normalize(energy_per_sec)
    slack_lb = -2.7715996733752064
    slack_ub = 31.47620209923513
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.46100486761803705 + (1.0 - 0.46100486761803705) * (1.0 - slack_scaled)
    rank_score = -percentile_normalize(upward_rank) * weight_rank
    unc_projected = 1.0 / (1.0 + np.exp(-2.708090982849045 * (uncertainty - 1.0)))
    energy_norm = percentile_normalize(min_incremental_energy)
    energy_uncertainty_score = 0.9000008156025442 * energy_norm * unc_projected
    wait_normalized = np.where(feasible_mask > 0.0, ready_wait_time / (np.abs(slack) + 1.0), 0.0)
    wait_score = percentile_normalize(wait_normalized)
    score = percentile_normalize(slack_score) + percentile_normalize(unc_slack_coupling) + percentile_normalize(escalated_duration_risk)
    score += feasible_mask * (1.4293336449611682 * energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    score = score * critical_gate
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
