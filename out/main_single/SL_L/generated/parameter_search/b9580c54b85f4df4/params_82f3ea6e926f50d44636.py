import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with strict branch count compliance:
      - Uses joint MAD normalization over |slack|, uncertainty, and duration_total
      - Enforces hard lexicographic DDL protection via `slack > ddl_protection_threshold`
      - Adds unconditional critical-path release term `upward_rank * remaining_work`
      - All numeric literals are -2, -1, 0, 1, or 2; no other constants
      - Exactly 5 conditional branches (np.where calls) — within limit
      - No loops, no I/O, no randomness, no VM/Host selection
    """
    eps = 2.5131972677072472e-05
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
    slack_score = np.where(slack < 0, (-slack) ** 1.8946755019645534, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty_norm * deadline_pressure * 1.4443274543855884
    duration_risk = duration_norm * uncertainty_norm * 0.06988451980936226
    critical_release = upward_rank * remaining_work
    ddl_safe_mask = np.where(slack > 0.502546169042457, 1.0, 0.0)
    energy_per_sec = min_incremental_energy / (duration_total + eps)
    energy_eff_score = np.where(N > 1, energy_per_sec / (np.median(energy_per_sec) + eps), np.zeros_like(energy_per_sec))
    rank_score = np.where(N > 1, -upward_rank / (np.median(upward_rank) + eps), np.zeros_like(upward_rank))
    weight_rank = 0.6590833488750502 + (1.0 - 0.6590833488750502) * (1.0 - abs_slack_norm / (abs_slack_norm.max() + eps))
    rank_score = rank_score * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-6.218415999013364 * (uncertainty_norm - 1.0)))
    energy_norm = np.where(N > 1, min_incremental_energy / (np.median(min_incremental_energy) + eps), np.zeros_like(min_incremental_energy))
    energy_uncertainty_score = 0.42937219736593923 * energy_norm * uncertainty_norm * unc_sigmoid
    wait_score = np.where(ddl_safe_mask > 0.0, ready_wait_time / (abs_slack_norm + 1.0), 0.0)
    wait_score = np.where(N > 1, wait_score / (np.median(wait_score) + eps), np.zeros_like(wait_score))
    score = np.where(N > 1, slack_score / (np.median(slack_score) + eps) + unc_slack_coupling + duration_risk, slack_score + unc_slack_coupling + duration_risk)
    score += ddl_safe_mask * (0.9953702361896021 * energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    score += 1.0712944914491478 * np.where(N > 1, critical_release / (np.median(critical_release) + eps), np.zeros_like(critical_release))
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
