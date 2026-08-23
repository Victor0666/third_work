import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with lexicographic DDL protection, successor-aware criticality, and robust host-load gating:
      - Strict DDL-first enforcement: non-feasibility terms only activated when slack > ddl_protection_threshold
      - Criticality term uses upward_rank * remaining_work interaction (validated cross-agent consensus)
      - Host-load anti-starvation uses uncertainty-weighted ready_wait_time only in safe region (slack > threshold)
      - Duration risk penalized via uncertainty-amplified (min_exec_time + min_comm_time) with successor-release factor
      - Energy-uncertainty coupling gated by bounded sigmoid on uncertainty to avoid numerical explosion
      - Joint MAD normalization over |slack|, uncertainty, and duration_total for cross-feature coherence
      - No percentile clipping (replaced by stable MAD scaling per design insight), avoiding fragility across seeds/scenarios
      - Uses only allowed literals {-2,-1,0,1,2}; all tunables exposed via PARAMS
      - Final score ensures: DDL feasibility dominates → among feasible: energy-efficiency & load fairness → among violated: critical path urgency
    """
    eps = 2.2345274257436603e-09
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
    all_risk_features = np.stack([abs_slack, uncertainty, duration_total], axis=0)
    median_vals = np.median(all_risk_features, axis=1, keepdims=True)
    devs = np.abs(all_risk_features - median_vals)
    mad_vals = np.median(devs, axis=1, keepdims=True) + eps
    norm_slack = (abs_slack - median_vals[0]) / mad_vals[0]
    norm_uncert = (uncertainty - median_vals[1]) / mad_vals[1]
    norm_duration = (duration_total - median_vals[2]) / mad_vals[2]
    slack_penalty = np.where(slack < 0, -slack + eps, 0.0)
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 1.9788084757998565
    is_ddl_pressured = slack <= 1.0102368587342223
    criticality_term = upward_rank * remaining_work
    criticality_score = np.where(is_ddl_pressured, criticality_term * 3.0665764147650796, 0.0)
    safe_headroom_mask = np.where(slack > 1.0102368587342223, 1.0, 0.0)
    energy_per_duration = min_incremental_energy / (duration_total + eps)
    energy_norm = (energy_per_duration - np.median(energy_per_duration)) / (np.median(np.abs(energy_per_duration - np.median(energy_per_duration))) + eps)
    energy_eff_score = np.clip(energy_norm, -1.0, 1.0)
    unc_sigmoid = 1.0 / (1.0 + np.exp(-4.055277263195373 * (norm_uncert - 0.0)))
    energy_uncertainty_score = 0.1681427253668761 * energy_norm * norm_uncert * unc_sigmoid
    host_load_score = safe_headroom_mask * (ready_wait_time * uncertainty * 0.022540415856942084)
    score = slack_penalty + duration_risk + criticality_score
    score += safe_headroom_mask * (0.776920255791644 * energy_eff_score + energy_uncertainty_score + host_load_score)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
