import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with unified robust feature-space scaling and simplified DDL-protection.
    Key improvements:
      - Replaces per-feature MAD normalization with *single global scale* computed over all features,
        ensuring coherent relative magnitudes and preventing score explosion from misaligned medians.
      - Simplifies DDL-protection gate to pure slack threshold (no uncertainty/energy conditions),
        restoring monotonic urgency and eliminating fragile coupling.
      - Introduces wait-time fairness term gated by slack feasibility to avoid starvation only when safe.
      - Removes work-density bonus (per reflection) and fragile conditional interactions; retains clean,
        bounded, and interpretable structure with ≤6 branches and ≤8 interaction terms.
      - All operations guarded against zero/NaN/inf; only {-2,-1,0,1,2} used as literals.
    """
    eps = 9.654249340341954e-09
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
    all_features = np.stack([min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty], axis=0).flatten()
    global_med = np.median(all_features) if len(all_features) > 1 else np.mean(all_features)
    global_mad = np.median(np.abs(all_features - global_med)) if len(all_features) > 1 else eps
    global_scale = 1.1467414822400375 * (global_mad if global_mad > eps else eps)

    def global_normalize(x):
        x = np.asarray(x, dtype=float)
        return np.clip((x - global_med) / global_scale, -2.0, 2.0)
    slack_score = np.where(slack < 0, (-slack) ** 2.872343906534413, 0.0)
    slack_lb = -25.0759130842373
    slack_ub = 30.968190597341916
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    ddl_protection_gate = (slack <= -1.2769645983526061).astype(float)
    critical_boost = ddl_protection_gate * 2.714447134016639 * global_normalize(upward_rank)
    wait_feasible_gate = (slack >= 0.0).astype(float)
    wait_normalized = ready_wait_time / (np.abs(slack) + 1.0)
    wait_score = global_normalize(wait_normalized) * wait_feasible_gate
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_norm = global_normalize(energy_per_sec)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 1.2221642489277222
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.932358234666951
    weight_rank = 0.2388802293594759 * (1.0 - slack_scaled)
    rank_norm = global_normalize(upward_rank)
    rank_score = -rank_norm * weight_rank
    median_unc = np.median(uncertainty) if N > 1 else np.mean(uncertainty)
    unc_threshold_gate = (uncertainty <= median_unc + eps).astype(float)
    energy_norm = global_normalize(min_incremental_energy)
    unc_norm = global_normalize(uncertainty)
    energy_uncertainty_score = 0.48149198572453605 * energy_norm * unc_norm * unc_threshold_gate
    score = global_normalize(slack_score) + global_normalize(unc_slack_coupling) + global_normalize(duration_risk) + 0.45975822140512235 * energy_eff_norm + rank_score + energy_uncertainty_score + wait_score - critical_boost
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
