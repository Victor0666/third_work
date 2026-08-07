import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with deadline proximity–uncertainty joint thresholding and adaptive urgency:
      - Uses percentile-based DDL-protection gating: activates bottleneck/energy only when
        (slack <= median_slack) AND (uncertainty >= percentile(uncertainty, uncertainty_percentile_threshold)).
      - Replaces all inactive parameters with single adaptive_urgency_scale modulating fuzzy urgency.
      - Uses smooth fuzzy deadline violation score: sigmoid((median_slack - slack) / (eps + std_slack)).
      - Energy term uses sigmoid-coupled uncertainty for bounded, interpretable modulation.
      - Fairness term applies wait_fairness_weight to inverted sigmoid of normalized wait time.
      - All normalizations use epsilon-guarded DDL-aware min-max; no rank-based nor std-only normalization.
      - Final composition: fuzzy_urgency > critical_path > energy_uncertainty > fairness.
    """
    eps = 1.7801192581634635e-06
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    finfo = np.finfo(float)
    median_slack = np.median(slack) if N > 0 else 0.0
    std_slack = np.std(slack) if N > 1 else eps
    urgency_input = (median_slack - slack) / (std_slack + eps)
    urgency_input_clipped = np.clip(urgency_input, -2.0, 2.0)
    fuzzy_urgency = 1.0 / (1.0 + np.exp(-urgency_input_clipped))
    std_uncertainty = np.std(uncertainty) if N > 1 else eps
    adaptive_urgency = fuzzy_urgency * (1.0 + 1.385244569427527 * std_uncertainty)
    u_min, u_max = (np.min(adaptive_urgency), np.max(adaptive_urgency))
    u_range = np.maximum(u_max - u_min, eps)
    norm_urgency = (adaptive_urgency - u_min) / (u_range + eps)
    q_unc = np.percentile(uncertainty, 78.23790639562466) if N > 0 else np.max(uncertainty)
    ddl_protection_active = (slack <= median_slack) & (uncertainty >= q_unc)
    critical_path_pressure = upward_rank * remaining_work
    cp_min, cp_max = (np.min(critical_path_pressure), np.max(critical_path_pressure))
    cp_range = np.maximum(cp_max - cp_min, eps)
    norm_critical_path = (critical_path_pressure - cp_min) / (cp_range + eps)
    unc_centered = uncertainty - np.median(uncertainty) if N > 0 else 0.0
    unc_std_safe = std_uncertainty + eps
    unc_sigmoid = 1.0 / (1.0 + np.exp(-unc_centered / unc_std_safe))
    energy_uncertainty_term = np.where(ddl_protection_active, min_incremental_energy * unc_sigmoid, 0.0)
    e_min, e_max = (np.min(energy_uncertainty_term), np.max(energy_uncertainty_term))
    e_range = np.maximum(e_max - e_min, eps)
    norm_energy_uncertain = np.where(ddl_protection_active, (energy_uncertainty_term - e_min) / (e_range + eps), 0.0)
    w_min, w_max = (np.min(ready_wait_time), np.max(ready_wait_time))
    w_range = np.maximum(w_max - w_min, eps)
    norm_wait_raw = (ready_wait_time - w_min) / (w_range + eps)
    wait_clipped = np.clip(norm_wait_raw, -2.0, 2.0)
    wait_sigmoid = 1.0 / (1.0 + np.exp(-wait_clipped))
    inv_wait = 1.0 - wait_sigmoid
    score = norm_urgency + 0.6032475868332303 * norm_critical_path + 0.5692760837316911 * norm_energy_uncertain - 0.4663247478273836 * inv_wait
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
