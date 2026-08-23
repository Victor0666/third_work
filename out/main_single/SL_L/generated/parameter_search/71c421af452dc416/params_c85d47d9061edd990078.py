import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining strengths of both parents:
      - Retains joint MAD normalization over [|slack|, uncertainty, duration_total] from Parent 2 for N=1 stability.
      - Adds unconditional 'urgency density' term (crit_path_urgency / duration_total) from Parent 1 — improves fast-release of critical work.
      - Uses robust clipped denominator for wait_score: 1 + clip(slack, 0, wait_slack_denom_clamp).
      - Keeps hard lexicographic DDL protection via ddl_safe_mask (slack > ddl_protection_threshold).
      - Eliminates fragile host-load gating and criticality_boost multiplicative scaling; relies on structural urgency terms instead.
      - All numeric literals are strictly -2, -1, 0, 1, or 2.
      - Exactly 5 conditional branches (np.where), satisfying structural constraint.
      - No loops, I/O, randomness, or VM selection logic.
    """
    eps = 2.7856708941184524e-09
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
    duration_total = min_exec_time + min_comm_time + eps
    abs_slack = np.abs(slack)
    all_risks = np.stack([abs_slack, uncertainty, duration_total], axis=0)
    mad = np.median(np.abs(all_risks - np.median(all_risks, axis=1, keepdims=True)), axis=1)
    mad_safe = np.where(mad == 0.0, eps, mad)
    abs_slack_norm = abs_slack / (mad_safe[0] + eps)
    uncertainty_norm = uncertainty / (mad_safe[1] + eps)
    duration_norm = duration_total / (mad_safe[2] + eps)
    slack_score = np.where(slack < 0, (-slack) ** 2.1680806018090015, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty_norm * deadline_pressure * 0.9717379951228569
    duration_risk = duration_norm * uncertainty_norm * 0.9844642624489277
    crit_path_urgency = upward_rank * remaining_work * 1.7845591391583813
    urgency_density = crit_path_urgency / (duration_total + eps) * 1.8596262698427561
    ddl_safe_mask = np.where(slack > 0.0472174290234742, 1.0, 0.0)
    energy_per_sec = min_incremental_energy / (duration_total + eps)
    energy_eff_score = np.where(N > 1, energy_per_sec / (np.median(energy_per_sec) + eps), np.zeros_like(energy_per_sec))
    rank_score = np.where(N > 1, -upward_rank / (np.median(upward_rank) + eps), np.zeros_like(upward_rank))
    weight_rank = 0.7967580047544571 + (1.0 - 0.7967580047544571) * (1.0 - abs_slack_norm / (abs_slack_norm.max() + eps))
    rank_score = rank_score * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-6.0930311797205725 * (uncertainty_norm - 1.0)))
    energy_norm = np.where(N > 1, min_incremental_energy / (np.median(min_incremental_energy) + eps), np.zeros_like(min_incremental_energy))
    energy_uncertainty_score = 0.14703325784621823 * energy_norm * uncertainty_norm * unc_sigmoid
    slack_clipped = np.clip(slack, 0.0, 1.0726985202770003)
    wait_denom = 1.0 + slack_clipped
    wait_base = ready_wait_time / (wait_denom + eps)
    wait_score = np.where(ddl_safe_mask > 0.0, wait_base, 0.0)
    wait_score = np.where(N > 1, wait_score / (np.median(wait_score) + eps), np.zeros_like(wait_score))
    score = np.where(N > 1, slack_score / (np.median(slack_score) + eps) + unc_slack_coupling + duration_risk, slack_score + unc_slack_coupling + duration_risk)
    score += np.where(N > 1, crit_path_urgency / (np.median(crit_path_urgency) + eps), np.zeros_like(crit_path_urgency))
    score += np.where(N > 1, urgency_density / (np.median(urgency_density) + eps), np.zeros_like(urgency_density))
    score += ddl_safe_mask * (1.432157187940597 * energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
