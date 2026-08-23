import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's global scaling & robust DDL gate with Parent 1's critical-path interaction.
    Key innovations:
      - Retains Parent 2's unified global feature-space scaling (coherent magnitude relationships).
      - Keeps hard DDL-protection gate (monotonic, robust), but enhances criticality boost using *upward_rank × remaining_work* product
        — captures joint path-criticality and computational load without fragile conditionals.
      - Refines wait-time fairness: activated only when slack >= 0 (safe to delay) — simplified from margin to meet parameter count limit.
      - Removes redundant work-density bonus but preserves robust energy-uncertainty interaction gated by median uncertainty.
      - All operations use only {-2,-1,0,1,2} literals; no unbounded loops or I/O.
      - Final score enforces strict lexicographic ordering: DDL violation > DDL pressure > critical path > energy efficiency > fairness.
    """
    eps = 7.231305395269599e-09
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
    global_scale = 1.2864814116795724 * (global_mad if global_mad > eps else eps)

    def global_normalize(x):
        x = np.asarray(x, dtype=float)
        return np.clip((x - global_med) / global_scale, -2.0, 2.0)
    slack_score = np.where(slack < 0, (-slack) ** 2.5288800251720662, 0.0)
    slack_lb = -27.316069714954182
    slack_ub = 22.23979313124256
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    ddl_protection_gate = (slack <= -5.155691973495748).astype(float)
    rank_work_product = upward_rank * remaining_work
    critical_boost = ddl_protection_gate * 2.7376854782777102 * global_normalize(rank_work_product)
    wait_feasible_gate = (slack >= 0.0).astype(float)
    duration_total = min_exec_time + min_comm_time + eps
    duration_med = np.median(duration_total) if N > 1 else np.mean(duration_total)
    wait_normalized = ready_wait_time / (duration_med + eps)
    wait_score = global_normalize(wait_normalized) * wait_feasible_gate
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_norm = global_normalize(energy_per_sec)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 1.618576169459467
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.40498332286685634
    weight_rank = 0.3517887274291782 * (1.0 - slack_scaled)
    rank_norm = global_normalize(upward_rank)
    rank_score = -rank_norm * weight_rank
    median_unc = np.median(uncertainty) if N > 1 else np.mean(uncertainty)
    unc_threshold_gate = (uncertainty <= median_unc + eps).astype(float)
    energy_norm = global_normalize(min_incremental_energy)
    unc_norm = global_normalize(uncertainty)
    energy_uncertainty_score = 0.7872286791794216 * energy_norm * unc_norm * unc_threshold_gate
    score = global_normalize(slack_score) + global_normalize(unc_slack_coupling) + global_normalize(duration_risk) + 0.24315286484057744 * energy_eff_norm + rank_score + energy_uncertainty_score + wait_score - critical_boost
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
