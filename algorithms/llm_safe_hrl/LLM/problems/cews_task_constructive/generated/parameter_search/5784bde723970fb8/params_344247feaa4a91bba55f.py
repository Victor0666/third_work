import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's structural strengths with Parent 1's robust uncertainty-energy coupling:
      - Retains Parent 2's DDL-aware min-max normalization (superior under small-N/tight-deadline stress).
      - Keeps conditional DDL-protection gating: activates bottleneck only when slack <= median_slack AND normalized uncertainty > threshold.
      - Adds Parent 1's uncertainty-modulated energy term, now gated by the same DDL-protection condition for coherence.
      - Uses bounded sigmoid wait saturation (Parent 2) but applies it to *normalized* wait time to preserve fairness across scales.
      - Replaces raw neg_slack with piecewise linear urgency ramp (Parent 1) for smoother gradient near deadline boundary.
      - All normalizations use epsilon-guarded DDL-aware min-max; no rank-based normalization to avoid rank distortion under skew.
      - Final score composition prioritizes: urgency > critical path > bottleneck > energy uncertainty > anti-starvation.
    """
    eps = 1.0322620878004574e-06
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
    tight_slack_boundary = median_slack * (1.0 - 0.48908974042324)
    urgency_linear = np.clip(tight_slack_boundary - slack, 0.0, None)
    urgency_mask = (slack <= tight_slack_boundary) & (slack >= -finfo.tiny)
    urgency = np.where(urgency_mask, urgency_linear, 0.0)
    u_min, u_max = (np.min(urgency), np.max(urgency))
    u_range = np.maximum(u_max - u_min, eps)
    norm_urgency = (urgency - u_min) / (u_range + eps)
    unc_std = np.std(uncertainty) if N > 1 else eps
    dispersion_base = 1.4979876035356066 * (unc_std + eps)
    unc_min, unc_max = (np.min(uncertainty), np.max(uncertainty))
    unc_range = np.maximum(unc_max - unc_min, eps)
    unc_normalized = (uncertainty - unc_min) / (unc_range + eps)
    ddl_protection_active = (slack <= median_slack) & (unc_normalized > 0.5012116359597256)
    critical_path_pressure = upward_rank * remaining_work
    cp_min, cp_max = (np.min(critical_path_pressure), np.max(critical_path_pressure))
    cp_range = np.maximum(cp_max - cp_min, eps)
    norm_critical_path = (critical_path_pressure - cp_min) / (cp_range + eps)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    bottleneck_base = duration * upward_rank * remaining_work
    bottleneck_pressure = np.where(ddl_protection_active, bottleneck_base * np.power(1.0 + unc_normalized, 0.9507988754216471), 0.0)
    bp_min, bp_max = (np.min(bottleneck_pressure), np.max(bottleneck_pressure))
    bp_range = np.maximum(bp_max - bp_min, eps)
    norm_bottleneck = np.where(ddl_protection_active, (bottleneck_pressure - bp_min) / (bp_range + eps), 0.0)
    energy_uncertainty_term = np.where(ddl_protection_active, min_incremental_energy * unc_normalized, 0.0)
    e_min, e_max = (np.min(energy_uncertainty_term), np.max(energy_uncertainty_term))
    e_range = np.maximum(e_max - e_min, eps)
    norm_energy_uncertain = np.where(ddl_protection_active, (energy_uncertainty_term - e_min) / (e_range + eps), 0.0)
    w_min, w_max = (np.min(ready_wait_time), np.max(ready_wait_time))
    w_range = np.maximum(w_max - w_min, eps)
    norm_wait = (ready_wait_time - w_min) / (w_range + eps)
    wait_clipped = np.clip(norm_wait, -2.0, 2.0)
    wait_sigmoid = 1.0 / (1.0 + np.exp(-wait_clipped))
    wait_saturation = np.power(wait_sigmoid, 0.8696555660627864)
    inv_wait = 1.0 - wait_saturation
    score = norm_urgency + 1.0536852125069511 * norm_critical_path + norm_bottleneck + 0.8566193272036552 * norm_energy_uncertain - inv_wait
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
